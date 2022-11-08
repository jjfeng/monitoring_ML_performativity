"""
Generate clinician for treatment propensities
"""
import sys, os
import argparse
import pickle
import logging

import numpy as np

from clinician import Clinician, ClinicianMultSurv


def parse_args():
    parser = argparse.ArgumentParser(description="Generate data for simulations")
    parser.add_argument(
        "--pre-propensity-beta",
        type=str,
    )
    parser.add_argument(
        "--pre-propensity-beta-unmeas",
        type=str,
    )
    parser.add_argument(
        "--shift-propensity-beta",
        type=str,
    )
    parser.add_argument(
        "--shift-propensity-beta-unmeas",
        type=str,
    )
    parser.add_argument(
        "--shift-scale", type=str, choices=["logit", "log_risk"]
    )
    parser.add_argument(
        "--shift-time",
        type=int,
        default=np.inf,
        help="Time when the propensity model does a single shift",
    )
    parser.add_argument("--out-file", type=str, default="_output/clinician.pkl")
    args = parser.parse_args()
    args.pre_propensity_beta = np.array(
        list(map(float, args.pre_propensity_beta.split(",")))
    ).reshape((-1, 1))

    if args.pre_propensity_beta_unmeas:
        args.pre_propensity_beta_unmeas = np.array(
            list(map(float, args.pre_propensity_beta_unmeas.split(",")))
        ).reshape((-1, 1))
    else:
        args.pre_propensity_beta_unmeas = np.zeros(args.pre_propensity_beta.shape)

    if args.shift_propensity_beta is None:
        args.shift_propensity_beta = np.zeros(args.pre_propensity_beta.shape)
    else:
        args.shift_propensity_beta = np.array(
            list(map(float, args.shift_propensity_beta.split(",")))
        ).reshape((-1, 1))

    if args.shift_propensity_beta_unmeas is None:
        args.shift_propensity_beta_unmeas = np.zeros(args.pre_propensity_beta_unmeas.shape)
    else:
        args.shift_propensity_beta_unmeas = np.array(
            list(map(float, args.shift_propensity_beta_unmeas.split(",")))
        ).reshape((-1, 1))

    assert args.pre_propensity_beta.size == args.shift_propensity_beta.size
    assert args.pre_propensity_beta_unmeas.size == args.shift_propensity_beta_unmeas.size
    assert args.pre_propensity_beta_unmeas.size == args.pre_propensity_beta.size
    return args

def main():
    args = parse_args()

    if args.shift_scale == "logit":
        clinician = Clinician(
            args.pre_propensity_beta,
            args.shift_propensity_beta,
            shift_scale=args.shift_scale,
            propensity_shift_time=args.shift_time,
        )
    elif args.shift_scale == "log_risk":
        clinician = ClinicianMultSurv(
            args.pre_propensity_beta,
            args.pre_propensity_beta_unmeas,
            args.shift_propensity_beta,
            args.shift_propensity_beta_unmeas,
            shift_scale=args.shift_scale,
            propensity_shift_time=args.shift_time,
        )
    with open(args.out_file, "wb") as f:
        pickle.dump(clinician, f)


if __name__ == "__main__":
    main()
