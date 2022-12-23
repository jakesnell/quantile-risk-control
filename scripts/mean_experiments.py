import math
import numpy as np
import scipy
from scipy.stats import ksone, ks_1samp, norm
from scipy.special import betaincinv
from scipy.optimize import bisect
import pandas as pd
from dataclasses import asdict, dataclass
from torch.utils.data import TensorDataset
import crossprob
import torch
from bisect import bisect_left
import argparse
from argparse import Namespace
import os
import pickle as pkl

from var_control.methods import (
    DKW,
    OrderStats,
    LttHB,
    RcpsWSR
)
from var_control.metric import sensitivity, balanced_accuracy, weighted_loss
from var_control.split import even_split_iterator, fixed_split_iterator
from var_control.utils import *
from var_control.bounds import *

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12., 5.]


def main(args):
    
    output_dir = "mean_output/{}".format(args.dataset)
    os.makedirs(output_dir, exist_ok = True)

    torch.manual_seed(args.seed)

    bound_list = [
        Bound("KS", ks_bound),
        Bound("BJ", berk_jones),
    ] 

    if args.loss == "balanced_accuracy":
        loss_fn = lambda pred, y: 1 - balanced_accuracy(pred, y)
        loss_type = "general"
        methods = [
            LttHB,
            RcpsWSR
        ]
    elif args.loss == "hamming_distance":
        
        def hamming(pred, y):
            pred = pred.permute(1,0,2)
            loss = torch.count_nonzero(pred != y, axis=-1)/pred.shape[-1]
            return loss.permute(1,0)
        
        loss_fn = lambda pred, y: hamming(pred, y)
        loss_type = "general"
        methods = [
            LttHB,
            RcpsWSR
        ]
    else:
        raise ValueError(f"Unexpected loss {args.loss}")

    save_dir = f"output/{args.dataset}_experiments"

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
    elif args.dataset == "go_emotions":
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
    
    save_string = "{}_{}_no_data_{}".format(args.dataset, args.loss, n_val_datapoints)
    print(save_string)

    method_dict = OrderedDict([(method.__name__, method) for method in methods])
    trial_results = []
    trial_idx = 0

    for train_sample, test_sample in tqdm(
            fixed_split_iterator(TensorDataset(preds.sum(-1), loss), args.num_trials, n_val_datapoints),
            total=args.num_trials,
        ):
            trial_idx += 1

            train_batch = Batch([x.t() for x in train_sample], args.num_hypotheses)
            test_batch = Batch([x.t() for x in test_sample], args.num_hypotheses)
            X = train_batch.loss
            correction = X.shape[0]
            n = X.shape[-1]

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

                aucs = integrate_quantiles(X, b, beta_min=0.0, beta_max=1.0)
                hyp_ind = np.argmin(aucs)
                auc = np.min(aucs)

                test_loss = test_batch.loss[hyp_ind]
                mean_loss = np.mean(test_loss.numpy())

                trial_results.append((trial_idx, bound_name, auc, mean_loss))

            for method_name, method in method_dict.items():

                if hasattr(method, "fit_risk"):
                    hyp_ind, mean_alpha = method.fit_risk(X, 1.0, args.delta)
                else:
                    raise NotImplementedError
                    
                test_loss = test_batch.loss[hyp_ind]
                mean_loss = np.mean(test_loss.numpy())

                trial_results.append((trial_idx, method_name, mean_alpha, mean_loss))

    results_df = pd.DataFrame(trial_results, columns=["trial", "method", "alpha", "mean loss"])
    average_df = results_df.drop(columns="trial").groupby(["method"]).mean()
    if args.save_csv:
        print("saving df to csv...")
        results_df.to_csv("{}/{}_full_results.csv".format(output_dir, save_string))
        average_df.to_csv("{}/{}.csv".format(output_dir, save_string))
        args_dict = vars(args)
        with open("{}/{}.pkl".format(output_dir, save_string), "wb") as handle:
            pkl.dump(args_dict, handle)
        print(args_dict)
    print(average_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mean bounding experiments")
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
        "--num_val_datapoints",
        type=int,
        default=500,
        help="number of validation points",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=100,
        help="size of beta grid",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="acceptable probability of error (default: 0.05)",
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
        action="store_true",
        help="store results"
    )
    parser.add_argument(
        "--dataset",
        default="coco",
        help="dataset for experiments"
    )
    args = parser.parse_args()
    main(args)
