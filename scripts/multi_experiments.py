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
)
from var_control.metric import sensitivity, balanced_accuracy, weighted_loss
from var_control.split import even_split_iterator, fixed_split_iterator
from var_control.utils import *
from var_control.bounds import *

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12., 5.]


def main(args):
    
    output_dir = "multi_output/{}".format(args.dataset)
    os.makedirs(output_dir, exist_ok = True)
    
    beta_lo = args.beta_lo
    beta_hi = args.beta_hi

    torch.manual_seed(args.seed)

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
        ]
    elif args.loss == "hamming_distance":
        
        def hamming(pred, y):
            pred = pred.permute(1,0,2)
            loss = torch.count_nonzero(pred != y, axis=-1)/pred.shape[-1]
            return loss.permute(1,0)
        
        loss_fn = lambda pred, y: hamming(pred, y)
        loss_type = "general"
        methods = [
            DKW,
            OrderStats,
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

    beta_grid_size = args.grid_size
    tolerance = -1e-6
    plot = False
    
    save_string = "{}_{}_beta_lo_{}_beta_hi_{}_no_data_{}_grid_size_{}".format(
        args.dataset, args.loss, beta_lo, beta_hi, n_val_datapoints, beta_grid_size
    )
    print(save_string)

    method_dict = OrderedDict([(method.__name__, method) for method in methods])
    trial_results = []
    trial_idx = 0

    for train_sample, test_sample in tqdm(
            fixed_split_iterator(TensorDataset(preds.sum(-1), loss), args.num_trials, n_val_datapoints),
            total=args.num_trials,
        ):
            trial_idx += 1
            if trial_idx == args.num_trials:
                plot = True

            train_batch = Batch([x.t() for x in train_sample], args.num_hypotheses)
            test_batch = Batch([x.t() for x in test_sample], args.num_hypotheses)
            X = train_batch.loss
            correction = X.shape[0]
            n = X.shape[-1]
            
            if plot:
                fig, (ax1, ax2) = plt.subplots(1, 2)

            for bound_item in bound_list:

                bound_name = bound_item.name
                bound_fn = bound_item.bound_fn

                if bound_item.b is not None:
                    b = bound_item.b
                else:
                    if bound_item.p_ignore:
                        b = bound_fn(n, args.delta/correction, beta_min=bound_item.p_ignore)
                    else:
                        b = bound_fn(n, args.delta/correction)
                    bound_item.b = b
                X_sorted = np.sort(X, axis=-1)
                
                def get_trial_result(test_batch, hyp_ind, beta, beta_lo, beta_hi, b):
                    test_loss = test_batch.loss[hyp_ind].numpy()
                    mean_loss = np.mean(test_loss)
                    quantile_loss = np.quantile(test_loss, args.beta).item()
                    
                    test_cdf = ecdf(X_sorted[hyp_ind], test_loss)
                    test_cdf = torch.Tensor(np.expand_dims(test_cdf, 0))
                    test_int_auc = integrate_quantiles(
                        torch.Tensor(np.expand_dims(test_loss, 0)),
                        np.arange(1, test_loss.shape[-1] + 1) / test_loss.shape[-1], beta_min=beta_lo, beta_max=beta_hi
                    )[0]
                    test_cvar_auc = integrate_quantiles(
                        torch.Tensor(np.expand_dims(test_loss, 0)),
                        np.arange(1, test_loss.shape[-1] + 1) / test_loss.shape[-1], beta_min=beta, beta_max=1.0
                    )[0]
                    return [mean_loss, quantile_loss, test_int_auc, test_cvar_auc]
                
                ##########################################
                ### Interval
                ##########################################
                aucs = integrate_quantiles(X, b, beta_min=beta_lo, beta_max=beta_hi)
                hyp_ind = np.argmin(aucs)
                auc = np.min(aucs)

                interval_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=beta_lo, beta_max=beta_hi)[0]
                cvar_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=args.beta, beta_max=1.0)[0]
                quantile_alpha = X_sorted[hyp_ind, (b < args.beta).astype(int).sum()]
                mean_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=0.0, beta_max=1.0)[0]
                
                assert np.isclose(auc, interval_auc), "interval: {} does not equal {}".format(auc, interval_auc)
                guarantees = [mean_auc, quantile_alpha, interval_auc, cvar_auc]

                trial_result = [trial_idx, bound_name, "Interval"] + guarantees + get_trial_result(
                    test_batch, hyp_ind, args.beta, beta_lo, beta_hi, b
                )
                trial_results.append(trial_result)
                
                ##########################################
                ### CVar
                ##########################################
                aucs = integrate_quantiles(X, b, beta_min=args.beta, beta_max=1.0)
                hyp_ind = np.argmin(aucs)
                auc = np.min(aucs)

                interval_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=beta_lo, beta_max=beta_hi)[0]
                cvar_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=args.beta, beta_max=1.0)[0]
                quantile_alpha = X_sorted[hyp_ind, (b < args.beta).astype(int).sum()]
                mean_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=0.0, beta_max=1.0)[0]
                
                assert np.isclose(auc, cvar_auc), "cvar: {} does not equal {}".format(auc, cvar_auc)
                guarantees = [mean_auc, quantile_alpha, interval_auc, cvar_auc]

                trial_result = [trial_idx, bound_name, "CVaR"] + guarantees + get_trial_result(
                    test_batch, hyp_ind, args.beta, beta_lo, beta_hi, b
                )
                trial_results.append(trial_result)
                
                ##########################################
                ### Mean
                ##########################################
                aucs = integrate_quantiles(X, b, beta_min=0.0, beta_max=1.0)
                hyp_ind = np.argmin(aucs)
                auc = np.min(aucs)
                
                interval_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=beta_lo, beta_max=beta_hi)[0]
                cvar_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=args.beta, beta_max=1.0)[0]
                quantile_alpha = X_sorted[hyp_ind, (b < args.beta).astype(int).sum()]
                mean_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=0.0, beta_max=1.0)[0]
                
                assert np.isclose(auc, mean_auc), "mean: {} does not equal {}".format(auc, mean_auc)
                guarantees = [mean_auc, quantile_alpha, interval_auc, cvar_auc]

                trial_result = [trial_idx, bound_name, "Mean"] + guarantees + get_trial_result(
                    test_batch, hyp_ind, args.beta, beta_lo, beta_hi, b
                )
                trial_results.append(trial_result)

                ##########################################
                ### VaR
                ##########################################
                quantile_alphas = X_sorted[:, (b < args.beta).astype(int).sum()]
                hyp_ind = np.argmin(quantile_alphas)
                alpha = np.min(quantile_alphas)
                
                interval_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=beta_lo, beta_max=beta_hi)[0]
                cvar_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=args.beta, beta_max=1.0)[0]
                quantile_alpha = X_sorted[hyp_ind, (b < args.beta).astype(int).sum()]
                mean_auc = integrate_quantiles(X[hyp_ind].unsqueeze(0), b, beta_min=0.0, beta_max=1.0)[0]
                
                assert np.isclose(alpha, quantile_alpha), "quantile: {} does not equal {}".format(alpha, quantile_alpha)
                guarantees = [mean_auc, quantile_alpha, interval_auc, cvar_auc]

                trial_result = [trial_idx, bound_name, "VaR"] + guarantees + get_trial_result(
                    test_batch, hyp_ind, args.beta, beta_lo, beta_hi, b
                )
                trial_results.append(trial_result)
                

    results_df = pd.DataFrame(trial_results, columns=[
        "trial", "method", "optimizer",
        "mean_guar", "var_guar", "interval_guar", "cvar_guar",
        "mean_loss", "var_loss", "interval_auc", "cvar_auc"
    ])
    average_df = results_df.drop(columns="trial").groupby(["method", "optimizer"]).mean()
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
    parser = argparse.ArgumentParser(description="Run multi-experiments")
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
        help="number of validation points (default: 1000)",
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
        "--beta",
        type=float,
        default=0.9,
        help="target quantile",
    )
    parser.add_argument(
        "--beta_lo",
        type=float,
        default=0.85,
        help="minimum quantile in interval",
    )
    parser.add_argument(
        "--beta_hi",
        type=float,
        default=0.95,
        help="maximum quantile in interval",
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
