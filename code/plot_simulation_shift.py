#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the distribution of alarm times")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--monitor-time", type=int)
    parser.add_argument("--shift-time", type=int)
    parser.add_argument("--alarm-rate", type=float)
    parser.add_argument("--result-files", type=str)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--csv-file", type=str, default="_output/res.csv")
    parser.add_argument("--plot-file", type=str, default="_output/shift_alarm.png")
    args = parser.parse_args()
    args.result_files = args.result_files.split("+")

    if args.shift_time is None:
        args.shift_time = args.monitor_time + 1

    return args


def plot_alert_times(
    alert_times,
    method_str,
    preshift_T=None,
    end_time=None,
    batch_size=1,
    false_alarm_rate=0.1,
    file_name="_output/alert_times.png",
):
    plt.clf()
    sns.ecdfplot(
        [t if np.isfinite(t) else end_time * 2 for t in alert_times],
    )

    # plot shift time
    if preshift_T is not None:
        plt.axvline(x=preshift_T, color="black", linestyle="--")
        detect_delay = np.median(
            [
                a - preshift_T
                for a in alert_times
                if (np.isfinite(a) and a >= preshift_T)
            ]
            + [end_time] * np.sum(np.isnan(alert_times))
        )
        plt.title("%s (median delay %.1f)" % (method_str, detect_delay))

    plt.axhline(y=false_alarm_rate, color="red", linestyle="--")

    plt.xlim(0, end_time + 1)
    plt.savefig(file_name)


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    logging.info("Number of replicates: %d", len(args.result_files))

    agg_alert_tot_times = []
    nonmissing_delays = []
    obs_cusums = []
    for idx, res_file in enumerate(args.result_files):
        if os.path.exists(res_file):
            res = pd.read_csv(res_file)
            method_str = res["method"][0]
            # print(res)
            obs_cusums.append(res["chart_obs"].iloc[-1])
            raw_alert_idxs = np.where(
                np.bitwise_or(
                    res["chart_obs"] > res["upper_thres"],
                    res["chart_obs"] < res["lower_thres"],
                )
            )[0]
            if raw_alert_idxs.size > 0:
                alert_tot_time = np.sum(
                    res["num_obs"].iloc[: np.min(raw_alert_idxs) + 1]
                )
                print("ALERT TOT TIME", alert_tot_time)
                if alert_tot_time >= args.shift_time:
                    alert_nonmissing_time = np.sum(
                        res["not_missings"].iloc[: np.min(raw_alert_idxs)]
                    )
                    nonmissing_shift_idxs = np.where(
                        np.cumsum(res["num_obs"]) > args.shift_time
                    )[0]
                    nonmissing_shift_time = (
                        np.sum(res["missings"].iloc[: np.min(nonmissing_shift_idxs)])
                        if nonmissing_shift_idxs.size
                        else 0
                    )
                    nonmissing_delays.append(
                        alert_nonmissing_time - nonmissing_shift_time
                    )
            else:
                alert_tot_time = np.NAN
            agg_alert_tot_times.append(alert_tot_time)
        else:
            print("file missing", res_file)

    # if args.batch_size == args.monitor_time:
    #    plt.hist(obs_cusums)
    #    plt.show()

    agg_alert_tot_times = np.array(agg_alert_tot_times)
    print(agg_alert_tot_times)
    good_alert_freq = np.mean(
        [np.isfinite(a) and (a >= args.shift_time) for a in agg_alert_tot_times]
    )
    logging.info("ALERT? %s", agg_alert_tot_times)
    logging.info("GOOD alarm rate %f", good_alert_freq)
    print("GOOD alarm rate %f", good_alert_freq)
    bad_alert_freq = np.mean(
        [np.isfinite(a) and (a < args.shift_time) for a in agg_alert_tot_times]
    )
    print(
        [a for a in agg_alert_tot_times if (np.isfinite(a) and (a < args.shift_time))]
    )
    logging.info("BAD alarm rate %f", bad_alert_freq)
    print("BAD alarm rate %f", bad_alert_freq)
    print(
        "BAD alarm rate (95 se) %f",
        1.96
        * np.sqrt(bad_alert_freq * (1 - bad_alert_freq) / agg_alert_tot_times.size),
    )
    logging.info(
        "average alert time %f",
        np.mean([a for a in agg_alert_tot_times if np.isfinite(a)]),
    )
    logging.info(
        "variance alert time %f",
        np.var([a for a in agg_alert_tot_times if np.isfinite(a)]),
    )
    logging.info(
        "std error alert time %f",
        np.sqrt(
            np.var([a for a in agg_alert_tot_times if np.isfinite(a)])
            / np.sum(np.isfinite(agg_alert_tot_times))
        ),
    )
    print(
        "avg alert time (among alerts)",
        np.mean([a for a in agg_alert_tot_times if np.isfinite(a)]),
    )
    print(
        "min avg alert time (censored)",
        np.mean(
            [a if np.isfinite(a) else args.monitor_time for a in agg_alert_tot_times]
        ),
    )
    print(
        "std error alert time",
        np.sqrt(
            np.var([a for a in agg_alert_tot_times if np.isfinite(a)])
            / np.sum(np.isfinite(agg_alert_tot_times))
        ),
    )

    plot_alert_times(
        agg_alert_tot_times,
        method_str,
        preshift_T=args.shift_time,
        end_time=args.monitor_time,
        false_alarm_rate=args.alarm_rate,
        batch_size=1,
        file_name=args.plot_file,
    )

    if args.shift_time <= args.monitor_time:
        detect_delay = np.mean(
            [
                a - args.shift_time
                for a in agg_alert_tot_times
                if np.isfinite(a) and a >= args.shift_time
            ]
        )
        print("DELAY total", detect_delay)
        logging.info("Detection delay (wrt total) %.2f", detect_delay)
        nonmissing_detect_delay = np.mean(nonmissing_delays)
        print("DELAY nonmissing", nonmissing_detect_delay)
        logging.info("Detection delay (wrt nonmissing) %.2f", nonmissing_detect_delay)

    alert_times = pd.DataFrame(
        {
            "alert_time": agg_alert_tot_times,
        }
    )
    alert_times["method"] = method_str
    alert_times.to_csv(args.csv_file, index=False)


if __name__ == "__main__":
    main()
