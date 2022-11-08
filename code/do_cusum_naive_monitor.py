"""
CUSUM naive monitoring (only for classification models)
"""
import time
import sys, os
import argparse
import logging
import pickle

from cusum import CusumNaive
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run CUSUM monitoring procedure")
    parser.add_argument(
        "--seed",
        type=int,
        default=1235,
        help="seed for determining meta-properties of the data",
    )
    parser.add_argument("--max-time", type=int, default=800)
    parser.add_argument("--n-calib", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--alarm-rate", type=float, default=0.1)
    parser.add_argument("--data-gen-file", type=str)
    parser.add_argument("--log-file", type=str, default="_output/monitor_log.txt")
    parser.add_argument("--out-chart-file", type=str, default="_output/res.csv")
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
    mtr = CusumNaive(
        data_gen,
        alpha=args.alarm_rate,
        max_time=args.max_time,
    )
    mtr.calibrate(n_calib=args.n_calib)

    st_time = time.time()
    hist_logger = mtr.monitor()
    run_time = time.time() - st_time
    print("alert time", hist_logger.alert_time)
    logging.info("alert time %s", hist_logger.alert_time)
    logging.info("run time %f", run_time)

    hist_logger.write_to_csv(args.out_chart_file)


if __name__ == "__main__":
    main()
