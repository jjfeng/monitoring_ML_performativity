#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import logging

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize chart statistic")
    parser.add_argument("--shift-time", type=int, default=None)
    parser.add_argument("--result-file", type=str, default="_output/res.csv")
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--chart-stat-plot", type=str, default="_output/chart_stat.png")
    args = parser.parse_args()
    return args

def plot_chart_stat(
    obs_chart_stats,
    chart_stat_upper_thres,
    num_obs,
    preshift_T=None,
    end_time=None,
    file_name="_output/test.png",
):
    times = np.cumsum(num_obs)
    data = pd.DataFrame(
        {
            "x": np.concatenate([times, times]),
            "chart_stat": np.concatenate([obs_chart_stats, chart_stat_upper_thres]),
            "label": ["Chart statistic"] * len(obs_chart_stats)
            + ["Control limit"] * len(obs_chart_stats),
        }
    )
    
    plt.clf()
    fig, ax = plt.subplots()
    a = sns.lineplot(x="x", y="chart_stat", hue="label", data=data, ax=ax)
    a.set_xlabel("Time")
    a.set_ylabel("Value")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    if preshift_T is not None:
        # Plot line of true shift time
        print("vline")
        plt.axvline(x=preshift_T, color="black", linestyle="--")

    if end_time:
        plt.xlim(0, end_time)
    sns.despine()
    plt.tight_layout()
    plt.savefig(file_name)

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    res = pd.read_csv(args.result_file)

    sns.set_context("paper", font_scale=1.8)

    plot_chart_stat(
        res.chart_obs,
        res.upper_thres,
        res.num_obs,
        preshift_T=args.shift_time,
        file_name=args.chart_stat_plot,
    )

if __name__ == "__main__":
    main()
