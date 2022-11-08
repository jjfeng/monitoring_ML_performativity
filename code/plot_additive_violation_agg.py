"""
Get additive violation?
"""
from importlib.util import module_for_loader
import time
import copy
import sys, os
import argparse
import logging
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression

import seaborn as sns
from matplotlib import pyplot as plt

from data_generator import DataGenerator
from generate_clinician import Clinician
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="measure equi-additivity violation")
    parser.add_argument(
        "--seed",
        type=int,
        default=1235,
        help="seed for determining meta-properties of the data",
    )
    parser.add_argument("--result-files", type=str, default="_output/res.csv")
    parser.add_argument("--shift-time", type=int)
    parser.add_argument("--labels", type=str, default="1,2,3")
    parser.add_argument("--out-csv-file", type=str, default="_output/violation.csv")
    parser.add_argument("--plot-file", type=str, default="_output/violation1.png")
    args = parser.parse_args()
    args.result_files = args.result_files.split("+")
    args.labels = args.labels.split(",")
    return args

def main():
    args = parse_args()
    res_df = []
    for label, res_f in zip(args.labels, args.result_files):
        df = pd.read_csv(res_f)
        df.loc[df.label == "observed", "label"] = label
        res_df.append(df)
    res_df = pd.concat(res_df).reset_index(drop=True)
    res_df = res_df.rename({
        'pred_risk': "Prediction",
        'cond_risk': "Conditional risk",
        'label': 'Setting',
        'time': 'Time',
    }, axis=1)
    print(res_df)

    plt.clf()
    sns.set_context("paper", font_scale=2)
    g = sns.relplot(
        x="Time",
        y="Conditional risk",
        style="Setting",
        hue="Prediction",
        data=res_df,
        kind="line",
    )
    plt.axvline(x=args.shift_time, color="black", linestyle="--")
    g.set(ylim=(0,1))
    plt.savefig(args.plot_file)

if __name__ == "__main__":
    main()
