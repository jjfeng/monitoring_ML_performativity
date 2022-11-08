"""
Bayesian monitoring
"""
import time
import sys, os
import argparse
import logging
import pickle
import json

import numpy as np

from generate_clinician import Clinician
from bayesian_monitor import BayesianMonitor
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bayesian changepoint monitoring")
    parser.add_argument(
        "--seed",
        type=int,
        default=1235,
        help="random seed",
    )
    parser.add_argument("--n-calib", type=int, default=0, help="Number of observations to use for initializing the Bayesian model")
    parser.add_argument("--confounder-start-idx", type=int, default=0, help="The start index of predictors used to stratify the model calibration curves")
    parser.add_argument("--confounder-end-idx", type=int, default=0, help="The end index of predictors used to stratify the model calibration curves")
    parser.add_argument("--max-time", type=int, default=200, help="Number of time points to run the procedure for")
    parser.add_argument("--alarm-rate", type=float, default=0.1, help="The desired false alarm probability")
    parser.add_argument(
        "--shift-scale",
        type=str,
        default="logit",
        choices=["logit", "risk"],
        help="What scale we are detecting a shift with respect to"
    )
    parser.add_argument("--num-integrate-samples", type=int, default=3000, help="Number of samples to integrate to estimate posterior distribution")
    parser.add_argument("--prior-shift-factor", type=float, default=1, help="Prior specifying the anticipated shift amount")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of observations to batch together")
    parser.add_argument("--data-gen-file", type=str, help="Data generator file")
    parser.add_argument("--no-halt", action="store_true", default=False, help="Whether to stop running the procedure if an alarm is fired")
    parser.add_argument("--temp-file", type=str, default="_output/tmp.json", help="Temporary file used to run CmdStan")
    parser.add_argument("--log-file", type=str, default="_output/monitor_log.txt", help="Log file")
    parser.add_argument("--out-chart-file", type=str, default="_output/res.csv", help="Output the chart statistic and control limits in this csv file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    with open(args.data_gen_file, "rb") as f:
        data_gen = pickle.load(f)

    data_gen.set_seed(args.seed)
    mtr = BayesianMonitor(
        data_gen,
        alpha=args.alarm_rate,
        prior_shift_factor=args.prior_shift_factor,
        tmp_data_path=args.temp_file,
        confounder_start_idx=args.confounder_start_idx,
        confounder_end_idx=args.confounder_end_idx,
        shift_scale=args.shift_scale,
        batch_size=args.batch_size,
        halt_when_alarm=not args.no_halt,
    )

    st_time = time.time()
    mtr.calibrate(n_calib=args.n_calib)
    hist_logger = mtr.monitor(args.max_time)
    run_time = time.time() - st_time
    print("alert time", hist_logger.alert_time)
    logging.info("alert time %s", hist_logger.alert_time)
    logging.info("run time %f", run_time)

    hist_logger.write_to_csv(args.out_chart_file)


if __name__ == "__main__":
    main()
