#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import datetime
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
    parser.add_argument("--alarm-rate", type=float)
    parser.add_argument("--result-files", type=str)
    parser.add_argument("--log-file", type=str, default="_output/log.txt")
    parser.add_argument("--csv-file", type=str, default="_output/res.csv")
    parser.add_argument("--plot-file", type=str, default="_output/shift_alarm.png")
    args = parser.parse_args()
    args.result_files = args.result_files.split("+")

    return args


def plot_alert_times(
    alert_times,
    method_str,
    batch_size=1,
    false_alarm_rate=0.1,
    file_name="_output/alert_times.png",
):
    plt.clf()

    end_time = datetime.date(2030,1,1)
    print(alert_times)
    alert_times = pd.to_datetime([end_time if t is None else t for t in alert_times])

    df = pd.DataFrame({"date": alert_times})
    counts_df = df.groupby([df["date"].dt.year, df["date"].dt.month]).count()
    percent_df = counts_df/len(alert_times)
    percent_df.plot(kind="bar")

    # sns.histplot(
    #     [end_time if t is None else t for t in alert_times],
    #     # cumulative=True,
    #     # stat="probability",
    #     binwidth=20,
    # )
    plt.gcf().autofmt_xdate()

    # plt.xlim(datetime.date(2020,1,1), datetime.date(2022,6,1))

    plt.savefig(file_name)


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    logging.info("Number of replicates: %d", len(args.result_files))

    agg_alert_tot_times = []
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
                alert_tot_time = datetime.datetime.strptime(res.time_labels[np.min(raw_alert_idxs)], "%Y-%m-%d").date()
                print("ALERT TOT TIME", alert_tot_time)
            else:
                alert_tot_time = None
            agg_alert_tot_times.append(alert_tot_time)
        else:
            print("file missing", res_file)

    plot_alert_times(
        agg_alert_tot_times,
        method_str,
        false_alarm_rate=args.alarm_rate,
        batch_size=1,
        file_name=args.plot_file,
    )

    alert_times = pd.DataFrame(
        {
            "alert_time": agg_alert_tot_times,
        }
    )
    alert_times["method"] = method_str
    alert_times.to_csv(args.csv_file, index=False)


if __name__ == "__main__":
    main()
