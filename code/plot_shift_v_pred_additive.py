#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import sys, os
import argparse
import pickle
import logging
from xml.etree.ElementPath import prepare_descendant

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression

from matplotlib import pyplot as plt
import seaborn as sns

from data_generator import DataGenerator
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Plot shift versus model prediction for observed propensities vs oracle propensities")
    parser.add_argument("--data-generator-file", type=str)
    parser.add_argument("--batch-size", type=int, default=40000)
    parser.add_argument("--plot", type=str, default="_output/shift_v_pred.png")
    args = parser.parse_args()
    return args

def get_pred_v_shift_obs(data_gen: DataGenerator, args, obs_factor=1):
    pre_data = data_gen.generate_data(n=args.batch_size * obs_factor, curr_time=0)
    pre_predictions = data_gen.mdl_dev.predict(pre_data.x)
    prechange_mdl = IsotonicRegression()
    prechange_mdl.fit(pre_predictions, pre_data.y.flatten())

    post_data = data_gen.generate_data(n=args.batch_size * obs_factor, curr_time=data_gen.shift_time + 1)
    post_predictions = data_gen.mdl_dev.predict(post_data.x)
    postchange_mdl = IsotonicRegression()
    postchange_mdl.fit(post_predictions, post_data.y.flatten())

    # eval_data = data_gen.generate_oracle_data(n=20 * args.batch_size * obs_factor, curr_time=data_gen.shift_time + 2)
    eval_data = data_gen.generate_oracle_data(n=20 * args.batch_size * obs_factor, curr_time=data_gen.shift_time - 2)
    eval_predictions = data_gen.mdl_dev.predict(eval_data.x)
    prechange_vals = prechange_mdl.predict(eval_predictions)
    postchange_vals = postchange_mdl.predict(eval_predictions)
    mdl_preds = 1/(1 + np.exp(-eval_predictions)).flatten()

    mask = eval_data.a == 0
    return mdl_preds[mask], (postchange_vals - prechange_vals)[mask]

def main():
    args = parse_args()
    with open(args.data_generator_file, "rb") as f:
        data_gen = pickle.load(f)
        data_gen.set_seed(0)

    obs_a0_mdl_pred, obs_shifts = get_pred_v_shift_obs(data_gen, args, obs_factor=2)

    oracle_data_gen = copy.copy(data_gen)
    oracle_data_gen.clinician = None
    oracle_data_gen.set_seed(0)
    oracle_a0_mdl_pred, oracle_shifts = get_pred_v_shift_obs(oracle_data_gen, args)

    sns.set_context("paper", font_scale=1.5)
    data1 = pd.DataFrame({
        "mdl_pred": obs_a0_mdl_pred,
        "risk_shift": obs_shifts,
    })
    data1["grp"] = "obs"
    data2 = pd.DataFrame({
        "mdl_pred": oracle_a0_mdl_pred,
        "risk_shift": oracle_shifts,
    })
    data2["grp"] = "oracle"
    print(data1.shape, data2.shape)
    data = pd.concat([data1, data2]).reset_index()
    sns.jointplot(data=data, x="mdl_pred", y="risk_shift", hue="grp") #, xlim=(0,1))
    plt.tight_layout()
    plt.savefig(args.plot)

if __name__ == "__main__":
    main()
