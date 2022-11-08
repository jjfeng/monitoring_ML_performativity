#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cProfile import label
import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression

from matplotlib import pyplot as plt
import seaborn as sns

from data_generator import DataGenerator
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Plot shift versus propensity")
    parser.add_argument("--data-generator-files", type=str)
    parser.add_argument("--label-title", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--max-time", type=int)
    parser.add_argument("--plot", type=str, default="_output/shift_v_prop.png")
    args = parser.parse_args()
    args.batch_size = 10000
    args.data_generator_files = args.data_generator_files.split("+")
    args.labels = args.labels.split(",")
    return args

def main():
    args = parse_args()
    all_df = []
    for label, data_gen_f in zip(args.labels, args.data_generator_files):
        with open(data_gen_f, "rb") as f:
            data_gen = pickle.load(f)
            data_gen.set_seed(0)

        eval_data = data_gen.generate_oracle_data(n=20 * args.batch_size, curr_time=data_gen.shift_time + 2)
        prechange_vals = data_gen.get_conditional_risk(eval_data.x, pre_beta=data_gen.pre_beta, curr_time=0)
        postchange_vals = data_gen.get_conditional_risk(eval_data.x, pre_beta=data_gen.pre_beta, curr_time=data_gen.shift_time + 2)
        a0_propensities = 1 - data_gen.clinician.get_propensities(eval_data.x, data_gen.mdl_dev, curr_time=data_gen.shift_time + 2)

        a0_mask = eval_data.a == 0
        propensity_mtr = a0_propensities[a0_mask]
        shift_mtr = (postchange_vals - prechange_vals)[a0_mask]
        iso_reg = IsotonicRegression()
        iso_reg.fit(propensity_mtr, shift_mtr)
        df = pd.DataFrame({
            "propensity": propensity_mtr,
            "shift_risk": iso_reg.predict(propensity_mtr)
        })
        df[args.label_title] = label
        all_df.append(df)
    all_df = pd.concat(all_df).reset_index(drop=True)

    sns.set_context("paper", font_scale=1.8)
    h = sns.jointplot(
        data=all_df,
        x="propensity",
        y="shift_risk",
        hue=args.label_title)
    h.set_axis_labels("Sampling propensity", "Shift in conditional risk")
    plt.tight_layout()
    plt.savefig(args.plot)


if __name__ == "__main__":
    main()
