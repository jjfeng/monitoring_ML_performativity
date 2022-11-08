#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from matplotlib import pyplot as plt
from dataset import Dataset
import seaborn as sns

from data_generator import DataGenerator
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Plot true risk vs model prediction (initially vs time of alarm)")
    parser.add_argument("--data-generator-file", type=str)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--scale", type=str, default="risk")
    parser.add_argument("--retrained-mdls-file", type=str, default=None)
    parser.add_argument("--plot", type=str, default="_output/risk_v_pred.png")
    args = parser.parse_args()
    return args

def get_mdl_eval_data(data_gen, mdl_dev, test_size, curr_time):
    eval_data = data_gen.generate_data(n=test_size, curr_time=curr_time)
    mdl_logits = mdl_dev.predict(eval_data.x).flatten()

    true_risks = data_gen.get_conditional_risk(eval_data.x, pre_beta=data_gen.pre_beta, curr_time=curr_time)
    true_logits = np.log(true_risks/(1 - true_risks))
    iso_reg = IsotonicRegression()
    iso_reg.fit(mdl_logits, true_logits)
    
    mdl_logit_inputs = np.arange(mdl_logits.min(), mdl_logits.max(), step=0.01)
    data = pd.DataFrame({
        "mdl_logit": mdl_logit_inputs,
        "true_logit": iso_reg.predict(mdl_logit_inputs).flatten(),
    })
    return data

def main():
    args = parse_args()
    with open(args.data_generator_file, "rb") as f:
        data_gen = pickle.load(f)
        data_gen.set_seed(0)

    with open(args.retrained_mdls_file, "rb") as f:
        retrained_mdls = pickle.load(f)

    dfs = []
    for mdl_res_dict in retrained_mdls:
        curr_time = mdl_res_dict["time"]
        locked_eval_df = get_mdl_eval_data(data_gen, data_gen.mdl_dev, args.test_size, curr_time)
        locked_eval_df["grp"] = curr_time//50 * 50
        locked_eval_df["label"] = "Locked"
        retrained_eval_df = get_mdl_eval_data(data_gen, mdl_res_dict["mdl"], args.test_size, curr_time)
        retrained_eval_df["grp"] = curr_time//50 * 50
        retrained_eval_df["label"] = "Retrained"
        dfs += [locked_eval_df, retrained_eval_df]

    data = pd.concat(dfs).reset_index()
    sns.set_context("paper", font_scale=1.5)
    data["mdl_risk"] = 1/(1 + np.exp(-data.mdl_logit))
    data["true_risk"] = 1/(1 + np.exp(-data.true_logit))

    sns.lmplot(data=data, x="mdl_risk", y="true_risk", col="label", hue="grp", scatter=False, lowess=True, palette="flare")

    plt.tight_layout()
    plt.savefig(args.plot)

if __name__ == "__main__":
    main()
