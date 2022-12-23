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
import os

from var_control.nursery_features import load_features_and_ground_truth, get_label_dict
from var_control.methods import (
    VarControl,
    RcpsBinomial,
    DKW,
    OrderStats,
    Inflation,
    LttHB
)
from var_control.metric import sensitivity, balanced_accuracy, weighted_loss, custom_loss
from var_control.split import even_split_iterator
from var_control.utils import *


def main(args):

    torch.manual_seed(args.seed)
    
    output_dir = "var_output/{}".format(args.dataset)
    os.makedirs(output_dir, exist_ok = True)

    if args.loss == "balanced_accuracy":
        loss_fn = lambda pred, y: 1 - balanced_accuracy(pred, y)
        loss_type = "general"
        methods = [
            DKW,
            OrderStats,
            LttHB,
        ]
    elif args.loss == "custom_loss":
        loss_fn = custom_loss
        loss_type = "general"
        methods = [
            DKW,
            OrderStats,
            LttHB,
        ]
    else:
        raise ValueError(f"Unexpected loss {args.loss}")

    z_p, y_p, z_n, y_n = load_features_and_ground_truth("logreg")
    label_dict = get_label_dict()

    thresholds = torch.linspace(
        min(z_p.min().item(), z_n.min().item()),
        max(z_p.max().item(), z_n.max().item()),
        args.num_hypotheses
    )

    split_sets = [
        (z_n, y_n, label_dict["finance"][0]),
        (z_p, y_p, label_dict["finance"][1]),
    ]

    for z_group, y_group, group in split_sets:

        preds = predict(thresholds, z_group).permute(1, 0, 2)  # [N, H, C] (long)
        if args.loss == "balanced_accuracy":
            loss = loss_fn(preds, y_group.unsqueeze(1))
        else:
            loss = loss_fn(preds, y_group)

        method_dict = OrderedDict([(method.__name__, method) for method in methods])

        trial_results = []
        oracle_vars, oracle_set_sizes = [], []
        for train_sample, test_sample in tqdm(
                even_split_iterator(TensorDataset(preds.sum(-1), loss), args.num_trials),
                total=args.num_trials,
        ):
            train_batch = Batch([x.t() for x in train_sample], args.num_hypotheses)
            test_batch = Batch([x.t() for x in test_sample], args.num_hypotheses)

            if loss_type == "general":
                oracle_hypothesis_ind, oracle_var = calculate_oracle(test_batch.loss, args.beta, "general")
            else:
                oracle_hypothesis_ind, oracle_var = calculate_oracle(test_batch.loss, args.beta, "monotonic",                                                args.alpha)
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

        rows = []
        for trial_ind, trial_result in enumerate(trial_results):
            for k, v in trial_result.items():
                rows.append({"trial": trial_ind + 1, "method": k} | asdict(v))

        results_df = pd.DataFrame(rows)
        print("Group results:", group)
        avg_results_df = results_df.drop(columns="trial").groupby(["method"]).mean()
        avg_results_df['oracle_var'] = sum(oracle_vars) / len(oracle_vars)
        avg_results_df['oracle_quantile_set_size'] = sum(oracle_set_sizes) / len(oracle_set_sizes)
        if args.save_csv:
            print("saving...")
            avg_results_df.to_csv(f"{output_dir}/avg_results_{args.loss}_{group}.csv")
            results_df.to_csv(f"{output_dir}/results_{args.loss}_{group}.csv")
            
        print(avg_results_df)
        if args.show_latex:
            print(avg_results_df.to_latex(float_format="%.3f"))

        if args.show_std:
            std_results_df = results_df.drop(columns="trial").groupby(["method"]).std()
            print(std_results_df)
            if args.show_latex:
                print(std_results_df.to_latex(float_format="%.3f"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run nursery var experiments")
    parser.add_argument(
        "loss", type=str, help="Loss function (options: weighted_loss, balanced_accuracy"
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
        "--alpha", type=float, default=0.2, help="target loss value (default: 0.2)"
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
        "--dataset",
        default="nursery",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--show_std",
        action="store_true",
        help="display standard deviation results"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="store results"
    )
    args = parser.parse_args()
    main(args)
