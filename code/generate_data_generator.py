"""
Wraps around everything to make data
"""
import sys, os
import argparse
import pickle

import numpy as np

from data_generator import DataGenerator
from generate_clinician import Clinician, ClinicianMultSurv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make data generator (this is the hospital)"
    )
    parser.add_argument("--pre-beta", type=str, help="coef and intercept prechange")
    parser.add_argument(
        "--shift-scale", type=str, default="logit", choices=["logit", "risk"]
    )
    parser.add_argument(
        "--shift-beta", type=str, help="shift parameterized by coef and intercept"
    )
    parser.add_argument(
        "--train-data-rate-monitor",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train-data-rate-calib",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--shift-time",
        type=int,
        default=np.inf,
    )
    parser.add_argument(
        "--shift-gradual-factor",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--do-unbiased-training",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--model-developer-file", type=str, default="_output/mdl_dev.pkl"
    )
    parser.add_argument("--clinician-file", type=str, default="_output/clinician.pkl")
    parser.add_argument("--out-file", type=str, default="_output/hospital.pkl")
    args = parser.parse_args()
    args.pre_beta = np.array(list(map(float, args.pre_beta.split(","))))
    if args.shift_beta is not None:
        args.shift_beta = np.array(list(map(float, args.shift_beta.split(","))))
    else:
        args.shift_beta = np.zeros(args.pre_beta.shape)
    # assert args.pre_beta.size == args.shift_beta.size
    return args


def main():
    args = parse_args()

    with open(args.model_developer_file, "rb") as f:
        mdl_dev = pickle.load(f)
    with open(args.clinician_file, "rb") as f:
        clinician = pickle.load(f)
    train_sampler = None if args.do_unbiased_training else clinician 

    data_gen = DataGenerator(
        args.pre_beta.reshape((-1,1)),
        args.shift_beta.reshape((-1,1)),
        args.shift_time,
        clinician,
        train_sampler=train_sampler,
        train_data_rate_monitor=args.train_data_rate_monitor,
        train_data_rate_calib=args.train_data_rate_calib,
        mdl_dev=mdl_dev,
        shift_scale=args.shift_scale,
        shift_time_factor=args.shift_gradual_factor,
    )
    with open(args.out_file, "wb") as f:
        pickle.dump(data_gen, f)


if __name__ == "__main__":
    main()
