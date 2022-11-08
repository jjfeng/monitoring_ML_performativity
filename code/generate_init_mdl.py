"""
Generate initial data, Train an initial model
"""
import sys, os
import argparse
import pickle
import logging

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt

from data_generator import DataGenerator
from model_developers import ModelDeveloperClassify, ModelDeveloperRisk, MyCalibratedClassifer
from common import *


def parse_args():
    parser = argparse.ArgumentParser(description="Generate data for simulations")
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
    )
    parser.add_argument("--max-features", type=int, default=1)
    parser.add_argument(
        "--pre-beta",
        type=str,
        default="5",
        help="coefs and intercept (last elem) for orig model",
    )
    parser.add_argument("--refit-freq", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--max-trains", type=str, default="0")
    parser.add_argument("--family", type=str, default="binomial")
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--do-classification", action="store_true", default=False)
    parser.add_argument(
        "--n-train",
        type=int,
        default=100000,
        help="how much data was used to train the initial model",
    )
    parser.add_argument("--out-mdl-file", type=str, default="_output/init_mdl.pkl")
    parser.add_argument("--roc-plot", type=str, default=None)
    parser.add_argument("--calibration-plot", type=str, default=None)
    parser.add_argument("--log-file", type=str, default="_output/init_mdl_log.txt")
    args = parser.parse_args()
    args.pre_beta = list(map(float, args.pre_beta.split(",")))
    args.max_trains = list(map(int, args.max_trains.split(",")))
    return args

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)

    true_yx_beta = np.array(args.pre_beta).reshape((-1, 1))

    data_gen = DataGenerator(
        pre_beta=true_yx_beta,
        shift_beta=None,
        shift_time=np.inf,
        clinician=None,
        mdl_dev=None,
        family=args.family,
    )
    data_gen.set_seed(args.seed)
    data = data_gen.generate_data(args.n_train, curr_time=0)
    print("treatment rate", data.a.mean())

    # orig prediction model
    if args.family == "bernoulli":
        if args.model_type == "logistic":
            orig_mdl = LogisticRegression(penalty="none", max_iter=10000, warm_start=True)
        elif args.model_type == "xgb":
            orig_mdl = MyCalibratedClassifer(ml_class=GradientBoostingClassifier, test_size=args.test_size, n_estimators=50, max_depth=1)
        elif args.model_type == "ridge":
            # This hyperparameter C was pretuned. We can change this to LogisticRegressionCV as well.
            orig_mdl = LogisticRegression(max_iter=6000, penalty="l2", solver="lbfgs", C=0.1)
        else:
            raise NotImplementedError("Haven't implemented this type of model developer")
    else:
        raise NotImplementedError("Haven't implemented this type of model developer")

    if args.do_classification:
        mdl_developer = ModelDeveloperClassify(
            orig_mdl,
            data,
            max_features=args.max_features,
            refit_freq=args.refit_freq,
            max_trains=args.max_trains,
        )
    else:
        mdl_developer = ModelDeveloperRisk(
            orig_mdl,
            data,
            max_features=args.max_features,
            refit_freq=args.refit_freq,
            max_trains=args.max_trains,
        )

    with open(args.out_mdl_file, "wb") as f:
        pickle.dump(mdl_developer, f)

    # if args.model_type == "ridge_nobin":
    #     logging.info("ORIG MDL CV-C %s", mdl_developer.mdls[0].C_)
    if args.model_type == "logistic":
        logging.info("ORIG MDL coef %s", mdl_developer.mdls[0].coef_)
        logging.info("ORIG MDL intercept %s", mdl_developer.mdls[0].intercept_)
    logging.info("ORIG MDL training sample size %d", np.sum(data.a.flatten() == 0))

    if args.family == "bernoulli":
        test_dat = data_gen.generate_data(50000, curr_time=2340)
        # CHECK MODEL AUC
        if args.do_classification:
            predictions = mdl_developer.predict(test_dat.x)
            ppv = np.mean(test_dat.y[predictions == 1])
            npv = 1 - np.mean(test_dat.y[predictions == 0])
            misclass_rate = np.mean(test_dat.y.flatten() != predictions)
            print("ppv npv misclass", ppv, npv, misclass_rate)
            logging.info("PPV %f", ppv)
            logging.info("NPV %f", npv)
            logging.info("misclass %f", misclass_rate)
        else:
            pred_logit = mdl_developer.predict(test_dat.x)
            auc = roc_auc_score(test_dat.y.flatten(), pred_logit)
            print("AUC", auc)
            logging.info("AUC %.3f", auc)
            if args.roc_plot is not None:
                from sklearn.metrics import RocCurveDisplay
                RocCurveDisplay.from_predictions(test_dat.y.flatten(), pred_logit.flatten())
                plt.savefig(args.roc_plot)

            if args.calibration_plot is not None:
                from sklearn.calibration import CalibrationDisplay
                pred_mu = 1/(1 + np.exp(-pred_logit))
                CalibrationDisplay.from_predictions(test_dat.y.flatten(), pred_mu.flatten())
                plt.savefig(args.calibration_plot)


if __name__ == "__main__":
    main()
