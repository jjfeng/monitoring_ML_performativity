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
    parser.add_argument("--propensity-shift-time", type=int, default=None)
    parser.add_argument("--max-time", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--risk-grid", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--data-gen-file", type=str)
    parser.add_argument("--log-file", type=str, default="_output/violation.txt")
    parser.add_argument("--out-csv-file", type=str, default="_output/violation.csv")
    parser.add_argument("--plot-file", type=str, default="_output/violation1.png")
    args = parser.parse_args()
    args.risk_grid = np.array(list(map(float, args.risk_grid.split(","))))
    return args


def get_conditional_exp(
    data_gen: DataGenerator, batch_size: int, curr_time: int, risk_grid: np.ndarray, eps: float = 0.01
):
    data = data_gen.generate_data(batch_size, curr_time=curr_time)
    mdl_x = data_gen.mdl_dev.select_features(data.x)
    true_risks = data_gen.get_conditional_risk(x=data.x, pre_beta=data_gen.pre_beta, curr_time=curr_time)
    pred_logit = data_gen.mdl_dev.predict(mdl_x)
    logging.info("HIST %s", np.histogram(pred_logit, bins=10))
    pred_prob = 1 / (1 + np.exp(-pred_logit))

    # Check how many observations were within this window
    print(np.array([np.sum(np.abs(pred_prob - p_risk) < eps) for p_risk in risk_grid]))

    return np.array([np.mean(true_risks[np.abs(pred_prob - p_risk) < eps]) for p_risk in risk_grid])


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    with open(args.data_gen_file, "rb") as f:
        data_gen = pickle.load(f)
    data_gen.set_seed(args.seed)

    oracle_data_gen = copy.copy(data_gen)
    oracle_data_gen.clinician = None
    oracle_data_gen.set_seed(args.seed)

    # measure violation of additive equi-confounding
    assert data_gen.mdl_dev.refit_freq is None
    all_dfs = []

    # Plot oracle first
    time_pts = [0, data_gen.shift_time + 1, args.max_time]
    for idx, t in enumerate(time_pts[:-1]):
        print("TIME", t)
        oracle_cond_exp = get_conditional_exp(
            data_gen=oracle_data_gen,
            batch_size=args.batch_size,
            curr_time=t,
            risk_grid=args.risk_grid,
        )
        for t_interim in range(t, time_pts[idx + 1]):
            cond_diff_df = pd.DataFrame(
                {
                    "pred_risk": (100 * args.risk_grid.flatten()).astype(int).astype(str),
                    "cond_risk": oracle_cond_exp,
                    "time": np.ones(args.risk_grid.size) * t_interim,
                }
            )
            cond_diff_df["label"] = "Oracle"
            all_dfs.append(cond_diff_df)

    # Plot observed conditional risks
    time_pts = [0, data_gen.shift_time + 1] + ([args.propensity_shift_time + 1] if args.propensity_shift_time is not None else []) + [args.max_time]
    for idx, t in enumerate(time_pts[:-1]):
        print("TIME", t)
        obs_cond_exp = get_conditional_exp(
            data_gen=data_gen,
            batch_size=args.batch_size,
            curr_time=t,
            risk_grid=args.risk_grid,
        )
        for t_interim in range(t, time_pts[idx + 1]):
            cond_diff_df = pd.DataFrame(
                {
                    "pred_risk": (100 * args.risk_grid.flatten()).astype(int).astype(str),
                    "cond_risk": obs_cond_exp,
                    "time": np.ones(args.risk_grid.size) * t_interim,
                }
            )
            cond_diff_df["label"] = "observed"
            all_dfs.append(cond_diff_df)

    res_df = pd.concat(all_dfs).reset_index()
    print(res_df)
    res_df.to_csv(args.out_csv_file, index=False)

    # make time-constant selection bias violation plot
    plt.clf()
    sns.set_context("paper", font_scale=2)
    sns.relplot(
        x="time",
        y="cond_risk",
        style="label",
        hue="pred_risk",
        data=res_df,
        kind="line",
    )
    plt.savefig(args.plot_file)

if __name__ == "__main__":
    main()
