#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay

from matplotlib import pyplot as plt
import seaborn as sns

from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize chart statistic")
    parser.add_argument("--calib-offset", type=int, default=None)
    parser.add_argument("--shift-time", type=int, default=None)
    parser.add_argument("--time-file", type=str, default="_output/data_time.csv")
    parser.add_argument("--result-file", type=str, default="_output/res.csv")
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--chart-stat-plot", type=str)
    parser.add_argument("--calibration-plot", type=str)
    parser.add_argument("--roc-plot", type=str)
    parser.add_argument("--out-csv", type=str, default="_output/chart_stat.csv")
    args = parser.parse_args()
    return args


def plot_chart_stat(
    obs_chart_stats,
    chart_stat_upper_thres,
    preshift_T=None,
    end_time=None,
    time_labels=None,
    file_name="_output/test.png",
):
    data = pd.DataFrame(
        {
            "x": np.concatenate([time_labels, time_labels]),
            "chart_stat": np.concatenate([obs_chart_stats, chart_stat_upper_thres]),
            "label": ["Chart statistic"] * len(obs_chart_stats)
            + ["Control limit"] * len(obs_chart_stats),
        }
    )
    data = data.groupby(by=["x", "label"]).max().reset_index()

    plt.clf()
    plt.figure(figsize=(6, 4))
    g = sns.lineplot(x="x", y="chart_stat", hue="label", data=data)
    g.legend_.set_title(None)
    g.set(ylabel="", xlabel="")
    sns.despine()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if end_time:
        plt.xlim(0, end_time)
    plt.savefig(file_name)

def plot_calibration(mdl_df, prechange_idx: int, alarm_idx: int, file_name: str, n_bins=5, strategy="quantile"):
    # Plot calibration pre-change
    pre_mask = np.arange(prechange_idx)
    changetime_str = mdl_df.datetime[prechange_idx].strftime("%Y-%m")
    pre_predictions = mdl_df.prediction[pre_mask]
    pre_prob_true, pre_prob_pred = calibration_curve(
        mdl_df.y[pre_mask], pre_predictions, n_bins=n_bins, strategy=strategy
    )
    # Plot calibration post-change
    post_mask = np.arange(prechange_idx, mdl_df.shape[0] if alarm_idx is None else alarm_idx)
    post_predictions = mdl_df.prediction[post_mask]
    post_prob_true, post_prob_pred = calibration_curve(
        mdl_df.y[post_mask], post_predictions, n_bins=n_bins, strategy=strategy
    )

    # Do plotting, with 80% confidence intervals
    _, ax = plt.subplots()
    pre_bin_size = pre_predictions.size/pre_prob_true.size
    ax.errorbar(pre_prob_pred, pre_prob_true, yerr=np.sqrt((1 - pre_prob_true) * pre_prob_true/pre_bin_size) * 1.282, fmt=".", capsize=3, label="Before %s" % changetime_str)
    post_bin_size = post_predictions.size/post_prob_true.size
    ax.errorbar(post_prob_pred, post_prob_true, yerr=np.sqrt((1 - post_prob_true) * post_prob_true/post_bin_size) * 1.282, fmt=".", capsize=3, label="After %s" % changetime_str)
    ax.plot([0, 1], [0, 1], "k:")
    ax.legend(loc="lower left")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    sns.despine()
    plt.tight_layout()
    plt.savefig(file_name)

def plot_roc(mdl_df, prechange_idx: int, alarm_idx: int, file_name: str, n_bins=5, strategy="quantile"):
    _, ax = plt.subplots()

    # Plot pre-change ROC
    pre_mask = np.arange(prechange_idx)
    changetime_str = mdl_df.datetime[prechange_idx].strftime("%Y-%m")
    pre_predictions = mdl_df.prediction[pre_mask]
    RocCurveDisplay.from_predictions(
            mdl_df.y[pre_mask],
            pre_predictions,
            name="Before %s" % changetime_str,
            ax=ax,
        )

    # Plot post-change ROC
    post_mask = np.arange(prechange_idx, mdl_df.shape[0] if alarm_idx is None else (alarm_idx + 1))
    post_predictions = mdl_df.prediction[post_mask]
    RocCurveDisplay.from_predictions(
            mdl_df.y[post_mask],
            post_predictions,
            name="After %s" % changetime_str,
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], "k:")
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    sns.despine()
    plt.tight_layout()
    plt.savefig(file_name)

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )

    sns.set_context("paper", font_scale=1.8)

    mdl_df = pd.read_csv(args.time_file)
    mdl_df.datetime = pd.to_datetime(mdl_df.datetime)
    raw_time_labels = mdl_df.datetime.to_numpy()

    res = pd.read_csv(args.result_file)
    time_labels = raw_time_labels[args.calib_offset + np.cumsum(res.num_obs).astype(int) - 1]
    res["time_labels"] = time_labels

    alarm_time = np.where(res.chart_obs > res.upper_thres)[0]
    alarm_time = alarm_time.min() if alarm_time.size else None
    print(alarm_time)
    if alarm_time is not None:
        logging.info("Cross time %s", time_labels[alarm_time])

        candidate_changepoint = res.candidate_changepoints[alarm_time]
        candidate_changepoint = np.sum(res.num_obs[:candidate_changepoint]) + args.calib_offset
        alarm_idx = np.sum(res.num_obs[:alarm_time]) + args.calib_offset
        logging.info("candidate change time %s", mdl_df.datetime[candidate_changepoint])
    else:
        alarm_idx = None
        candidate_changepoint = mdl_df.shape[0]//2
        logging.info("no alarm")

    plot_calibration(mdl_df, prechange_idx=candidate_changepoint, alarm_idx=alarm_idx, file_name=args.calibration_plot)
    # Note that this is AUC among the untreated patients, which has some weird dependency on the treatment propensity model
    plot_roc(mdl_df, prechange_idx=candidate_changepoint, alarm_idx=alarm_idx, file_name=args.roc_plot)

    res.to_csv(args.out_csv, index=False)

    if args.chart_stat_plot:
        plot_chart_stat(
            res.chart_obs,
            res.upper_thres,
            preshift_T=args.shift_time,
            time_labels=time_labels,
            file_name=args.chart_stat_plot,
        )



if __name__ == "__main__":
    main()
