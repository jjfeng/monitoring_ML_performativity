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
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--max-time", type=int, default=400)
    parser.add_argument("--plot-mod", type=int, default=50)
    parser.add_argument("--scale", type=str, default="risk")
    parser.add_argument("--mdls-file", type=str, default=None)
    parser.add_argument("--plot", type=str, default="_output/risk_v_pred.png")
    args = parser.parse_args()
    args.scale = args.scale.split(",")
    return args

def main():
    args = parse_args()
    with open(args.data_generator_file, "rb") as f:
        data_gen = pickle.load(f)
        data_gen.set_seed(0)
        # data_gen.clinician = None

    if args.mdls_file is not None and data_gen.mdl_dev.refit_freq is not None:
        with open(args.mdls_file, "rb") as f:
            mdls = pickle.load(f)
    else:
        mdls = [{
            "time": idx,
            "mdl": data_gen.mdl_dev,
        } for idx in range(0, args.max_time, args.batch_size)]

    dfs = []
    for mdl_res_dict in mdls:
        curr_time = mdl_res_dict["time"]
        if curr_time % args.plot_mod != 0:
            continue

        eval_data = data_gen.generate_data(n=args.test_size, curr_time=curr_time)
        mdl_logits = mdl_res_dict["mdl"].predict(eval_data.x).flatten()

        true_risks = data_gen.get_conditional_risk(eval_data.x, pre_beta=data_gen.pre_beta, curr_time=curr_time)
        true_logits = np.log(true_risks/(1 - true_risks))
        iso_reg = IsotonicRegression()
        iso_reg.fit(mdl_logits, true_logits)
        
        mdl_logit_inputs = np.arange(mdl_logits.min(), mdl_logits.max(), step=0.01)
        data_logit = pd.DataFrame({
            "Prediction": mdl_logit_inputs,
            "True": iso_reg.predict(mdl_logit_inputs).flatten(),
        })
        data_logit["Scale"] = "Logit"
        data_logit["Time"] = curr_time

        data_risk = data_logit.copy()
        data_risk["Scale"] = "Risk"
        data_risk["Prediction"] = 1/(1 + np.exp(-data_logit.Prediction))
        data_risk["True"] = 1/(1 + np.exp(-data_logit["True"]))
        if "risk" in args.scale:
            dfs.append(data_risk)
        if "logit" in args.scale:
            dfs.append(data_logit)

    data = pd.concat(dfs).reset_index()
    sns.set_context("paper", font_scale=1.5)

    # Make plot
    palette = sns.color_palette()
    seed_color = palette[1] if data_gen.mdl_dev.refit_freq is not None else palette[0]
    cmap = sns.dark_palette(seed_color, n_colors=data.Time.unique().size)
    sns.set_context("paper", font_scale=1.8)
    if len(args.scale) == 2:
        g = sns.lmplot(data=data, x="Prediction", y="True", col="Scale", hue="Time", scatter=False, lowess=True, palette=cmap, sharex=False, sharey=False, line_kws={"ls": "-", "alpha": 0.9})
    else:
        g = sns.lmplot(data=data, x="Prediction", y="True", hue="Time", scatter=False, lowess=True, palette=cmap, sharex=False, sharey=False, line_kws={"ls": "-", "alpha": 0.9})
    for col_val, ax in g.axes_dict.items():
        min_val = np.min(data.Prediction[data.Scale == col_val])
        max_val = np.max(data.Prediction[data.Scale == col_val])
        ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray")

    plt.savefig(args.plot)

if __name__ == "__main__":
    main()
