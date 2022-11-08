import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EPSILON = 1e-10
CLASSIFIERS = ["LogisticRegression", "RandomForestClassifier", "MLPClassifier"]


def logit_to_prob(logit):
    return 1 / (1 + np.exp(-logit))


def get_log_lik(prob, y):
    return np.log(prob) * y + np.log(1 - prob) * (1 - y)


def get_safe_logit(x, mdl):
    pred_prob = mdl.predict_proba(x)[:, 1:]
    pred_prob_safe = 0.999 * pred_prob + 0.001 * 0.5
    return np.log(pred_prob_safe / (1 - pred_prob_safe))


def reverse_cumsum(x):
    rev_cumsum = np.sum(x, axis=1, keepdims=True) - np.hstack(
        [
            np.zeros((x.shape[0], 1)),
            np.cumsum(x, axis=1)[:, :-1],
        ]
    )
    return rev_cumsum


def get_pred_logit(x, mdl):
    if mdl.__class__.__name__ in CLASSIFIERS:
        return get_safe_logit(x, mdl)
    elif mdl.__class__.__name__ == "LinearRegression":
        pred_prob = (1 / (1 + np.exp(-mdl.predict(x)))).reshape((-1, 1))
        return np.log(pred_prob / (1 - pred_prob))
    else:
        raise ValueError("mdl not know")


def create_recalib_inputs(x, orig_mdl, num_vars):
    if orig_mdl.__class__.__name__ in CLASSIFIERS:
        # pred_prob = orig_mdl.predict_proba(x)[:, 1:2]
        ml_logit = get_safe_logit(orig_mdl, x)
    elif orig_mdl.__class__.__name__ == "LinearRegression":
        pred_prob = (1 / (1 + np.exp(-orig_mdl.predict(x)))).reshape((-1, 1))
        ml_logit = np.log(pred_prob / (1 - pred_prob))
    else:
        raise NotImplementedError("dunno how to create propensity input")
    if num_vars == 2:
        prop_input_var = np.hstack([ml_logit, np.power(ml_logit, 2) * (ml_logit < 0)])
    elif num_vars == 1:
        prop_input_var = ml_logit
    else:
        raise NotImplementedError("only 2 inputs max right now")
    return prop_input_var


def create_mu_inputs(x, w, orig_mdl, num_ws, degree=2, n_knots=5):
    if orig_mdl.__class__.__name__ in CLASSIFIERS:
        # pred_prob = orig_mdl.predict_proba(x)[:, 1:2]
        ml_logit = get_safe_logit(orig_mdl, x)
    elif orig_mdl.__class__.__name__ == "LinearRegression":
        pred_prob = (1 / (1 + np.exp(-orig_mdl.predict(x)))).reshape((-1, 1))
        ml_logit = np.log(pred_prob / (1 - pred_prob))
    else:
        raise NotImplementedError("dunno how to create inputs")
    if num_ws:
        # mu_input_var = np.hstack([x, ml_logit, w[:, : num_vars - 1]])
        mu_input_var = np.hstack([x, w[:, :num_ws]])
    else:
        # mu_input_var = np.hstack([x, ml_logit])
        mu_input_var = x
    return mu_input_var
