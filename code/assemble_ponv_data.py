"""
Process MPOG PONV data
"""
import logging
import argparse
import pickle

import pandas as pd
import numpy as np

from data_generator import PreloadedDataGenerator
from dataset import Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="wrapper for PONV data"
    )
    parser.add_argument("--in-file", type=str, default="data/mpog_ponv/rf_ponv.csv")
    parser.add_argument("--outcome", type=str, default="y")
    parser.add_argument("--n-calib", type=int, default=100)
    parser.add_argument("--out-file", type=str, default="_output/ponv.pkl")
    parser.add_argument("--log-file", type=str, default="_output/ponv_log.txt")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    df = pd.read_csv(args.in_file)
    print("oall data", df.shape)
    print("after filtering", df[args.outcome].sum())

    data_x = df[["prediction"]].to_numpy()
    data_y = df[[args.outcome]].to_numpy()
    all_data = Dataset(
        data_x,
        np.zeros((data_x.shape[0], 1)),
        data_y
    )
    logging.info("all data %d", all_data.size)
    logging.info("all events %d", all_data.y.sum())

    assert all_data.size > args.n_calib
    data_gen = PreloadedDataGenerator(
        calib_data=all_data.subset(0, args.n_calib),
        monitor_data=all_data.subset(args.n_calib, all_data.size),
    )
    with open(args.out_file, "wb") as f:
        pickle.dump(data_gen, f)


if __name__ == "__main__":
    main()
