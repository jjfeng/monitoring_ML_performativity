#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate csv")
    parser.add_argument("--result-files", type=str)
    parser.add_argument("--shift-time", type=int)
    parser.add_argument("--out-tex", type=str, default="_output/out.tex")
    parser.add_argument("--out-csv", type=str, default="_output/out.csv")
    args = parser.parse_args()
    args.result_files = args.result_files.split("+")
    return args


def main():
    args = parse_args()

    all_res = []
    for idx, res_file in enumerate(args.result_files):
        res = pd.read_csv(res_file)
        all_res.append(res)
    all_res = pd.concat(all_res)

    assert args.shift_time is not None
    pre_shift_alert = all_res["alert_time"] <= args.shift_time
    all_res["pre_shift_alert"] = pre_shift_alert
    post_shift_alert = all_res["alert_time"] > args.shift_time
    all_res["post_shift_alert"] = post_shift_alert
    all_res.groupby("method").mean().to_csv(args.out_csv)
    print(all_res.groupby("method").mean())
    with open(args.out_tex, "w") as f:
        all_res.groupby("method").mean().to_latex(f)


if __name__ == "__main__":
    main()
