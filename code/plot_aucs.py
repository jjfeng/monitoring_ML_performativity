#!/usr/bin/env python
# -*- coding: utf-8 -*-

from operator import concat
import sys, os
import argparse
import pickle
from typing import List
import logging

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize chart statistic")
    parser.add_argument("--result-files", type=str)
    parser.add_argument("--plot-file", type=str, default="_output/aucs.csv")
    args = parser.parse_args()
    args.result_files = args.result_files.split("+")
    return args

def main():
    args = parse_args()
    auc_dfs = []
    for res_file in args.result_files:
        auc_dfs.append(pd.read_csv(res_file, index_col=0))
    all_auc_df = pd.concat(auc_dfs).reset_index(drop=True)

    sns.set_context("paper", font_scale=1.8)
    ax = sns.lineplot(x="time", y="auc", data=all_auc_df)
    ax.set_xlabel("Time")
    ax.set_ylabel("AUC")
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.plot_file)

if __name__ == "__main__":
    main()
