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
from var_control.metric import sensitivity, balanced_accuracy, weighted_loss, custom_loss
from var_control.split import even_split_iterator, fixed_split_iterator
from var_control.utils import *
from var_control.bounds import *

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12., 5.]


def main(args):
    
    output_dir = "pareto_output/{}".format(args.dataset)
    os.makedirs(output_dir, exist_ok = True)
    
    beta_lo = args.beta_lo
    beta_hi = args.beta_hi

    torch.manual_seed(args.seed)

    bound_list = [
        Bound("KS", ks_bound),
        Bound("BJ", berk_jones),
        Bound("One-sided-BJ", berk_jones_one_sided)
    ] 
    
    if beta_hi < 1.0:
        bound_list.append(
            Bound("Two-sided-BJ", berk_jones_two_sided)
        )

    if args.loss == "balanced_accuracy":
        loss_fn = lambda pred, y: 1 - balanced_accuracy(pred, y)
        loss_type = "general"
        methods = [
            DKW,
            OrderStats,
        ]
    elif args.loss == "custom_loss":
        loss_fn = custom_loss
        loss_type = "general"
        methods = [
            DKW,
            OrderStats,
        ]
    else:
        raise ValueError(f"Unexpected loss {args.loss}")

    save_dir = f"output/{args.dataset}_experiments"

    if args.dataset == "nursery":
        from var_control.nursery_features import load_features_and_ground_truth, get_label_dict
        feature_key = "logreg"
    else:
        raise ValueError

    z_p, y_p, z_n, y_n = load_features_and_ground_truth("logreg")
    label_dict = get_label_dict()

    n_val_datapoints = args.num_val_datapoints
    
    print("args:", args)
    print("no. datapoints:", z_p.shape[0]+z_n.shape[0])

    all_thresholds = torch.linspace(
        min(z_p.min().item(), z_n.min().item()),
        max(z_p.max().item(), z_n.max().item()),
        args.num_hypotheses
    )

    split_sets = [
        (z_n, y_n, label_dict["finance"][0]),
        (z_p, y_p, label_dict["finance"][1]),
    ]
    
    beta_grid_size = args.grid_size
    tolerance = -1e-6

    for z, y, group in split_sets:

        # make predictions on the entire set
        preds = predict(all_thresholds, z).permute(1, 0, 2)  # [N, H, C] (long)

        if args.loss in ["balanced_accuracy", "sensitivity"]:
            loss = loss_fn(preds, y.unsqueeze(1))
        else:
            loss = loss_fn(preds, y)
            
        if args.fixed_pred:
            aucs = integrate_quantiles(
                        loss.T,
                        np.arange(1, loss.shape[0] + 1) / loss.shape[0], 
                        beta_min=beta_lo, 
                        beta_max=beta_hi
                    )
            best_hyp = np.argmin(aucs)
            thresholds = all_thresholds[best_hyp:best_hyp+1]
            loss = loss[:, best_hyp:best_hyp+1]
            preds = preds[:, best_hyp:best_hyp+1]
            args.num_hypotheses = 1

        plot = False

        save_string = "{}_{}_{}_beta_lo_{}_beta_hi_{}_no_data_{}_grid_size_{}".format(args.dataset, args.loss, group, beta_lo, beta_hi, n_val_datapoints, beta_grid_size)
        if args.fixed_pred:
            save_string += "_fixed_pred"
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
                        if bound_name in ["KS", "BJ"]:
                            b = bound_fn(n, args.delta/correction)
                        elif bound_name == "One-sided-BJ":
                            b = bound_fn(n, args.delta/correction, q_min=beta_lo)
                        elif bound_name == "Two-sided-BJ":
                            b = bound_fn(n, args.delta/correction, q_min=beta_lo, q_max=beta_hi)
                        else:
                            raise ValueError
                        bound_item.b = b

                    aucs = integrate_quantiles(X, b, beta_min=beta_lo, beta_max=beta_hi)
                    hyp_ind = np.argmin(aucs)
                    auc = np.min(aucs)

                    test_loss = test_batch.loss[hyp_ind]
                    X_sorted = np.sort(X, axis=-1)
                    test_cdf = ecdf(X_sorted[hyp_ind], test_loss)
                    violation = int(np.any(
                        test_cdf < b,
                        axis=-1
                    ))
                    
                    test_int_auc = integrate_quantiles(
                        torch.Tensor(np.expand_dims(test_loss, 0)),
                        np.arange(1, test_loss.shape[-1] + 1) / test_loss.shape[-1], beta_min=beta_lo, beta_max=beta_hi
                    )[0]

                    trial_results.append((trial_idx, bound_name, auc, test_int_auc, violation))

                    if plot:
                        ax1.plot(b, X_sorted[hyp_ind], label=bound_name)
                        ax2.plot(X_sorted[hyp_ind], b, label=bound_name)
                        ax2.plot(X_sorted[hyp_ind], test_cdf, label=bound_name+"_test")


                for method_name, method in method_dict.items():

                    beta_vals = np.linspace(beta_lo, beta_hi, beta_grid_size)

                    if hasattr(method, "fit_front"):
                        bounded_region = method.fit_front(
                            train_batch.loss, args.delta/beta_grid_size, beta_vals
                        ).numpy()
                        mean_alpha = math.nan
                    else:
                        raise NotImplementedError

                    assert bounded_region.shape[-1] == beta_grid_size
                    assert bounded_region.max().item()<=1.0
                    aucs = integrate_quantiles(bounded_region, beta_vals, beta_min=beta_lo, beta_max=beta_hi)
                    hyp_ind = np.argmin(aucs)
                    auc = aucs[hyp_ind].item()

                    test_loss = test_batch.loss[hyp_ind]
                    test_cdf = ecdf(bounded_region[hyp_ind], test_loss)
                    violation = int(np.any(
                        test_cdf < beta_vals,
                        axis=-1
                    ))

                    test_int_auc = integrate_quantiles(
                        torch.Tensor(np.expand_dims(test_loss, 0)),
                        np.arange(1, test_loss.shape[-1] + 1) / test_loss.shape[-1], beta_min=beta_lo, beta_max=beta_hi
                    )[0]

                    trial_results.append((trial_idx, method_name+"Bonferroni", auc, test_int_auc, violation))

                    if plot:
                        ax1.plot(beta_vals, bounded_region[hyp_ind], label=method_name)
                        ax2.plot(bounded_region[hyp_ind], beta_vals, label=method_name)
                        ax2.plot(bounded_region[hyp_ind], test_cdf, label=method_name+"_test")


        results_df = pd.DataFrame(
            trial_results, 
            columns=["trial", "method", "guaranteed_auc", "empirical_auc", "lcb_violation"]
        )
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
    parser = argparse.ArgumentParser(description="Run nursery interval experiments")
    parser.add_argument(
        "loss", type=str, help="Loss function (options: weighted_accuracy)"
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
        help="number of validation datapoints",
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
        default="tiny-imagenet",
        help="dataset for experiments"
    )
    parser.add_argument(
        "--fixed_pred",
        action="store_true",
        help="fix predictor"
    )
    args = parser.parse_args()
    main(args)
