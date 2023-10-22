"""
CUSUM IPW monitoring (for brier score)
"""
import time
import sys, os
import argparse
import logging
import pickle

from matplotlib import pyplot as plt

from cusum import CusumIPW
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run IPW CUSUM monitoring procedure")
    parser.add_argument(
        "--seed",
        type=int,
        default=1235,
        help="seed for determining meta-properties of the data",
    )
    parser.add_argument("--n-calib", type=int, default=0, help="Number of observations to initialize the procedure")
    parser.add_argument("--confounder-start-idx", type=int, default=0, help="The start index of predictors used to stratify the model calibration curves")
    parser.add_argument("--confounder-end-idx", type=int, default=0, help="The end index of predictors used to stratify the model calibration curves")
    parser.add_argument("--max-time", type=int, default=200, help="Number of time points to run the procedure for")
    parser.add_argument("--max-look-back", type=int, default=0, help="Number of batches to consider for candidate changepoints (zero means consider all previous time points as potential changepoints)")
    parser.add_argument("--alarm-rate", type=float, default=0.1, help="Desired false alarm rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of observations to group into a single batch")
    parser.add_argument("--particles-per-batch", type=int, default=4, help="The number of bootstrap sequences will be equal to particles-per-batch * total number of batches.")
    parser.add_argument("--no-halt", action="store_true", default=False, help="Whether to stop running the procedure if an alarm is fired")
    parser.add_argument("--data-gen-file", type=str, help="Data generator file")
    parser.add_argument("--log-file", type=str, default="_output/monitor_log.txt", help="Log file")
    parser.add_argument("--out-chart-file", type=str, default="_output/res.csv", help="Output the chart statistic and control limits in this csv file")
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
    mtr = CusumIPW(
        data_gen,
        alpha=args.alarm_rate,
        confounder_start_idx=args.confounder_start_idx,
        confounder_end_idx=args.confounder_end_idx,
        batch_size=args.batch_size,
        num_particles=int(
            args.particles_per_batch / (args.alarm_rate / args.max_batch)
        ),
        halt_when_alarm=not args.no_halt,
    )
    mtr.calibrate(n_calib=10000)

    st_time = time.time()
    hist_logger = mtr.monitor(args.max_time)
    run_time = time.time() - st_time
    print("alert time", hist_logger.alert_time)
    logging.info("alert time %s", hist_logger.alert_time)
    logging.info("run time %f", run_time)

    hist_logger.write_to_csv(args.out_chart_file)
    
    # plt.plot(hist_logger.chart_statistics, label="ipw_chart")
    # plt.plot(hist_logger.control_limits, label="control")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
