"""
Sequential score-based CUSUM monitoring
"""
import time
import sys, os
import argparse
import logging
import pickle
import json

import numpy as np

from generate_clinician import Clinician
from score_monitor import ScoreMonitor
from common import *

NORM_DICT = {
    "L2": 2,
    "L1": 1,
    "inf": np.inf
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run score-based CUSUM monitoring with dynamic control limits to detect changes in model calibration (or stratified model calibration curves)")
    parser.add_argument(
        "--seed",
        type=int,
        default=1235,
        help="Random seed",
    )
    parser.add_argument("--n-calib", type=int, default=0, help="Number of observations to initialize the procedure")
    parser.add_argument("--confounder-start-idx", type=int, default=0, help="The start index of predictors used to stratify the model calibration curves")
    parser.add_argument("--confounder-end-idx", type=int, default=0, help="The end index of predictors used to stratify the model calibration curves")
    parser.add_argument("--max-time", type=int, default=200, help="Number of time points to run the procedure for")
    parser.add_argument("--max-look-back", type=int, default=0, help="Number of batches to consider for candidate changepoints (zero means consider all previous time points as potential changepoints)")
    parser.add_argument("--alarm-rate", type=float, default=0.1, help="Desired false alarm rate")
    parser.add_argument("--norm", type=str, default="L2", choices=["inf", "L1", "L2"], help="Norm used to summarize the cumulative score into a test statistic.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of observations to group into a single batch")
    parser.add_argument("--particles-per-batch", type=int, default=4, help="The number of bootstrap sequences will be equal to particles-per-batch * total number of batches.")
    parser.add_argument("--shift-scale", type=str, default="logit", choices=["logit", "risk"], help="What scale to detect shifts in model calibration")
    parser.add_argument("--no-halt", action="store_true", default=False, help="Whether to stop running the procedure if an alarm is fired")
    parser.add_argument("--data-gen-file", type=str, help="Data generator file")
    parser.add_argument("--log-file", type=str, default="_output/monitor_log.txt", help="Log file")
    parser.add_argument("--out-chart-file", type=str, default="_output/res.csv", help="Output the chart statistic and control limits in this csv file")
    parser.add_argument("--out-mdls-file", type=str, default="_output/mdls.pkl", help="Output the trained models in pickle files")
    args = parser.parse_args()
    args.max_batch = args.max_time // args.batch_size
    assert args.max_time % args.batch_size == 0
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
    mtr = ScoreMonitor(
        data_gen,
        norm=NORM_DICT[args.norm],
        alpha=args.alarm_rate,
        confounder_start_idx=args.confounder_start_idx,
        confounder_end_idx=args.confounder_end_idx,
        shift_scale=args.shift_scale,
        batch_size=args.batch_size,
        max_look_back=args.max_look_back,
        num_particles=int(
            args.particles_per_batch / (args.alarm_rate / args.max_batch)
        ),
        halt_when_alarm=not args.no_halt,
    )

    st_time = time.time()
    _, calib_mdl_hist = mtr.calibrate(n_calib=args.n_calib)
    hist_logger = mtr.monitor(args.max_time)
    run_time = time.time() - st_time
    print("alert time", hist_logger.alert_time)
    logging.info("alert time %s", hist_logger.alert_time)
    logging.info("run time %f", run_time)

    hist_logger.write_to_csv(args.out_chart_file)
    if args.out_mdls_file:
        hist_logger.mdl_history = calib_mdl_hist + hist_logger.mdl_history
        hist_logger.write_mdls(args.out_mdls_file)


if __name__ == "__main__":
    main()
