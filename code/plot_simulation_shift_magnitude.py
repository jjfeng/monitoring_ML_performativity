#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

LINEWIDTH = 2
LINE_STYLE_DICT = {
    'Locked': '-',
    'Retrain': '--',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot the distribution of alarm times, overlaid")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--monitor-times", type=str)
    parser.add_argument("--shift-time", type=int)
    parser.add_argument("--alarm-rate", type=float)
    parser.add_argument("--aggregate-files", type=str)
    parser.add_argument("--standardize-x", action="store_true", default=False)
    parser.add_argument("--title", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--plot-file", type=str, default="_output/shift_alarm.png")
    args = parser.parse_args()
    args.aggregate_files = args.aggregate_files.split("+")
    args.labels = args.labels.split(",")
    args.monitor_times = list(map(int, args.monitor_times.split(",")))

    return args


def plot_alert_times(
    alert_time_df,
    end_times,
    palette,
    standardize_x=False,
    preshift_T=None,
    false_alarm_rate=0.1,
    ax = None,
    ls: str = "-",
    legend: bool = False
):
    for label, end_time in zip(alert_time_df.label.unique(), end_times):
        if standardize_x:
            alert_times = [t/end_time if np.isfinite(t) else 2 for t in
                alert_time_df.alert_time[alert_time_df.label == label]]
        else:
            alert_times = [t if np.isfinite(t) else end_time * 2 for t in
                alert_time_df.alert_time[alert_time_df.label == label]]
        alert_time_df.loc[alert_time_df.label == label, "alert_time"] = alert_times
    
    print(alert_time_df)
    ax = sns.ecdfplot(
        data=alert_time_df,
        x="alert_time",
        hue="label",
        ax=ax,
        ls=ls,
        legend=legend,
        palette=palette,
    )

    if preshift_T is not None:
        # plot shift time
        plt.axvline(x=preshift_T, color="black", linestyle="--")
    else:
        # plot desired alert rate if no shift
        plt.axhline(y=false_alarm_rate, color="red", linestyle="--")

    if standardize_x:
        plt.xlim(0, 1)
        ax.set_xlabel(r"Relative alarm time (t/mK)")
    else:
        plt.xlim(0, end_time + 1)
        ax.set_xlabel("Alarm time")
    
    return ax


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    
    sns.set_context("paper", font_scale=1.8)

    agg_df = []
    for label, res_f in zip(args.labels, args.aggregate_files):
        if not os.path.exists(res_f):
            continue
        agg_alert_tot_times = pd.read_csv(res_f)
        agg_alert_tot_times["label"] = label.split("_")[0]
        agg_alert_tot_times["retrain"] = label.split("_")[1]
        agg_df.append(agg_alert_tot_times)
    agg_df = pd.concat(agg_df).reset_index(drop=True)
    
    palette = sns.color_palette()[:3]
    ax = None
    for retrain_status in agg_df.retrain.unique():
        filter_df = agg_df[agg_df.retrain == retrain_status]
        if retrain_status == "Retrain":
            filter_df = filter_df[filter_df.label != "ipw"]
        ax = plot_alert_times(
            filter_df,
            end_times=args.monitor_times,
            palette=palette,
            standardize_x=args.standardize_x,
            false_alarm_rate=args.alarm_rate,
            preshift_T=args.shift_time,
            ls=LINE_STYLE_DICT[retrain_status],
            ax = ax,
            legend=False
        )
    ax.set_title(args.title)

    lines = [
        Line2D([0], [0], color=c, linewidth=LINEWIDTH, linestyle='-') for c in palette] + [
        Line2D([0], [0], color='black', linewidth=LINEWIDTH, linestyle='-'),
        Line2D([0], [0], color='black', linewidth=LINEWIDTH, linestyle='--'),
        ]
    labels = ['ScoreCUSUM', 'Bayes', 'IPW CUSUM', 'Algo: Locked', 'Algo: Retrained']
    plt.legend(lines, labels) #, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    sns.despine()
    plt.savefig(args.plot_file)

if __name__ == "__main__":
    main()
