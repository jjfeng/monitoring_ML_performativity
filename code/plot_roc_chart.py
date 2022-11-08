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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

from matplotlib import pyplot as plt

from model_developers import ModelDeveloper
from data_generator import DataGenerator
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize chart statistic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-gen-file", type=str)
    parser.add_argument("--mdl-history-file", type=str)
    parser.add_argument("--out-auc-file", type=str, default="_output/aucs.csv")
    args = parser.parse_args()
    return args

def get_aucs(data_gen: DataGenerator, mdl_hist: List[ModelDeveloper], n_test: int = 4000):
    # Get AUCs
    all_aucs = []
    for mdl_idx, mdl_dict in enumerate(mdl_hist):
        test_dat = data_gen.generate_data(n_test, curr_time=mdl_dict["time"], factor=3)
        if mdl_dict["mdl"].mdls[0].__class__ == "MyLogisticRegressionCV":
            predictions = mdl_dict["mdl"].mdls[0].predict_raw_proba(test_dat.x)[:,1]
        else:
            predictions = mdl_dict["mdl"].mdls[0].predict_proba(test_dat.x)[:,1]
        # if mdl_idx == 0:
        #     a = RocCurveDisplay.from_predictions(test_dat.y, 1/(1 + np.exp(-predictions)))
        #     a.plot()
        #     plt.show()
        auc = roc_auc_score(test_dat.y.flatten(), predictions)
        all_aucs.append(pd.DataFrame({
            "time": [mdl_dict["time"]],
            "auc": [auc],
            }))
    # a = RocCurveDisplay.from_predictions(test_dat.y, 1/(1 + np.exp(-predictions)))
    # a.plot()
    # plt.show()
    return pd.concat(all_aucs)

def main():
    args = parse_args()
    with open(args.data_gen_file, "rb") as f:
        data_gen = pickle.load(f)
        data_gen.clinician = None
        data_gen.set_seed(args.seed)

    with open(args.mdl_history_file, "rb") as f:
        mdl_hist = pickle.load(f)
    auc_df = get_aucs(data_gen, mdl_hist)
    auc_df.to_csv(args.out_auc_file)

if __name__ == "__main__":
    main()
