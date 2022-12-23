import argparse
from argparse import Namespace
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Tuple, Optional
import math
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from scipy.stats import binom, norm
import scipy.stats
import pickle as pkl
import os

from var_control.methods import (
    VarControl,
    RcpsBinomial,
    DKW,
    OrderStats,
    Inflation,
    LttHB,
    RcpsHB
)
from var_control.metric import sensitivity, balanced_accuracy, weighted_loss
from var_control.split import even_split_iterator, fixed_split_iterator
from var_control.utils import *
from var_control.bounds import ks_bound, berk_jones, integrate_quantiles, Bound


def main(args: Namespace):

    torch.manual_seed(args.seed)
    
    output_dir = "var_output/{}".format(args.dataset)
    os.makedirs(output_dir, exist_ok = True)
    
    bound_list = [
        Bound("KS", ks_bound),
        Bound("BJ", berk_jones),
    ]

    if args.loss == "balanced_accuracy":
        loss_fn = lambda pred, y: 1 - balanced_accuracy(pred, y)
        loss_type = "general"
        methods = [
            DKW,
            OrderStats,
            LttHB,
        ]
    else:
        raise ValueError(f"Unexpected loss {args.loss}")

    if args.dataset in ["tiny-imagenet", "tiny-imagenet-mixed"]:
        from var_control.tiny_imagenet_features import load_features_and_ground_truth
        if args.dataset == "tiny-imagenet":
            feature_key = "resnet50"
        else:
            feature_key = "resnet50-mixed"
    elif args.dataset == "coco":
        from var_control.coco_features import load_features_and_ground_truth
        feature_key = "tresnet_m"
    elif args.dataset == "clip_cifar_100":
        from var_control.cifar_100_features import load_features_and_ground_truth
        feature_key = "clip_b32"
    elif args.dataset == "clip_cifar_100":
        from var_control.go_emotions_features import load_features_and_ground_truth
        feature_key = "bert"
    else:
        raise ValueError

    # y: [N, C]
    z, y = load_features_and_ground_truth(feature_key)
    
    n_val_datapoints = args.num_val_datapoints
    
    print("args:", args)
    print("no. datapoints:", z.shape[0])

    thresholds = torch.linspace(z.min().item(), z.max().item(), args.num_hypotheses)

    # make predictions on the entire set
    preds = predict(thresholds, z).permute(1, 0, 2)  # [N, H, C] (long)

    if args.loss in ["balanced_accuracy", "sensitivity"]:
        loss = loss_fn(preds, y.unsqueeze(1))
    else:
        loss = loss_fn(preds, y)

    method_dict = OrderedDict([(method.__name__, method) for method in methods])

    trial_results = []
    oracle_vars, oracle_set_sizes = [], []
    for train_sample, test_sample in tqdm(
        fixed_split_iterator(TensorDataset(preds.sum(-1), loss), args.num_trials, n_val_datapoints),
        total=args.num_trials,
    ):
        train_batch = Batch([x.t() for x in train_sample], args.num_hypotheses)
        test_batch = Batch([x.t() for x in test_sample], args.num_hypotheses)

        if loss_type == "general":
            oracle_hypothesis_ind, oracle_var = calculate_oracle(test_batch.loss, args.beta, "general")
        else:
            oracle_hypothesis_ind, oracle_var = calculate_oracle(test_batch.loss, args.beta, "monotonic", args.alpha)
        oracle_loss = test_batch.loss[oracle_hypothesis_ind]
        oracle_pred_size = test_batch.pred_size[oracle_hypothesis_ind]
        oracle_inds = quantile_indices(oracle_loss, args.beta)
        oracle_mean_in_quantile_pred_size = oracle_pred_size[oracle_inds].float().mean(-1).item()
        oracle_vars.append(oracle_var)
        oracle_set_sizes.append(oracle_mean_in_quantile_pred_size)

        if loss_type == "monotonic":
            trial_results.append(
                run_monotonic_trial(
                    train_batch,
                    test_batch,
                    method_dict,
                    alpha=args.alpha,
                    beta=args.beta,
                    delta=args.delta,
                    markov_scaling=args.markov_scaling,
                )
            )
        elif loss_type == "general":
            trial_results.append(
                run_general_trial(
                    train_batch,
                    test_batch,
                    method_dict,
                    alpha=args.alpha,
                    beta=args.beta,
                    delta=args.delta,
                    markov_scaling=args.markov_scaling,
                )
            )
        else:
            raise ValueError(f"Unexpected loss type {loss_type}")

        n = train_batch.loss.shape[-1]
        correction = args.num_hypotheses
        bound_res = OrderedDict()

        for bound_item in bound_list:

            bound_name = bound_item.name
            bound_fn = bound_item.bound_fn

            if bound_item.b is not None:
                b = bound_item.b
            else:
                if bound_name in ["KS", "BJ"]:
                    b = bound_fn(n, args.delta/correction)
                else:
                    raise ValueError
                bound_item.b = b
                
            X_sorted = np.sort(train_batch.loss, axis=-1)
            quantile_alphas = X_sorted[:, (b < args.beta).astype(int).sum()]
            hyp_ind = np.argmin(quantile_alphas)
            alpha = np.min(quantile_alphas)
            test_loss = test_batch.loss[hyp_ind].numpy()
            quantile_loss = np.quantile(test_loss, args.beta).item()
            res = GeneralTrialResult(
                quantile_loss=quantile_loss,
                quantile_alpha=alpha,
            )
            bound_res[bound_name]=res

        trial_results[-1].update(bound_res)

    rows = []
    for trial_ind, trial_result in enumerate(trial_results):
        for k, v in trial_result.items():
            rows.append({"trial": trial_ind + 1, "method": k} | asdict(v))

    results_df = pd.DataFrame(rows)
    avg_results_df = results_df.drop(columns="trial").groupby(["method"]).mean()
    avg_results_df['oracle_var'] = sum(oracle_vars)/len(oracle_vars)
    avg_results_df['oracle_quantile_set_size'] = sum(oracle_set_sizes) / len(oracle_set_sizes)
    
    if args.save_csv:
        print("saving...")
        results_df.to_csv(f"{output_dir}/results_{args.loss}.csv")
        avg_results_df.to_csv(f"{output_dir}/avg_results_{args.loss}.csv")
        args_dict = vars(args)
        with open(f"{output_dir}/{args.loss}.pkl", "wb") as handle:
            pkl.dump(args_dict, handle)
        print(args_dict)
        
    print(avg_results_df)
    if args.show_latex:
        print(avg_results_df.to_latex(float_format="%.3f"))

    if args.show_std:
        std_results_df = results_df.drop(columns="trial").groupby(["method"]).std()
        print(std_results_df)
        if args.show_latex:
            print(std_results_df.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VaR experiments")
    parser.add_argument(
        "loss", type=str, help="Loss function"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--num_hypotheses",
        type=int,
        default=500,
        help="number of hypotheses (default: 500)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1000,
        help="number of random splits (default: 1000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="target loss value (default: 0.25)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.9, help="target quantile level (default: 0.9)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="acceptable probability of error (default: 0.05)",
    )
    parser.add_argument(
        "--markov_scaling",
        action="store_true",
        help="use Markov's inequality to convert risk controlling to var controlling",
    )
    parser.add_argument(
        "--show_latex",
        action="store_true",
        help="display latex output"
    )
    parser.add_argument(
        "--show_std",
        action="store_true",
        help="display standard deviation results"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true"
    )
    parser.add_argument(
        "--dataset",
        default="tiny-imagenet",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--num_val_datapoints",
        type=int,
        default=500,
        help="number of validation datapoints",
    )
    args = parser.parse_args()
    main(args)
