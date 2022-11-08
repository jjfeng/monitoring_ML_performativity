import logging
from typing import List

import numpy as np
import sklearn.base
from sklearn.linear_model import LogisticRegression

from dataset import Dataset
from common import get_safe_logit

class MyCalibratedClassifer:
    classes_ = [0,1]

    def __init__(self, ml_class, test_size, *args, **kwargs):
        self.ml_class = ml_class
        self.test_size = test_size
        self.mdl = ml_class(*args, **kwargs)

    def fit(self, X, y):
        train_size = int(y.shape[0] - self.test_size)
        shuffled_indices = np.random.choice(y.shape[0], size=y.shape[0], replace=False)
        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]

        self.mdl.fit(X[train_indices], y[train_indices])
        pred_probs = self.mdl.predict_proba(X[test_indices])[:,1:]
        pred_odds = np.log(pred_probs/(1 - pred_probs))
        
        self.calib_mdl = LogisticRegression(penalty="none")
        self.calib_mdl.fit(pred_odds, y[test_indices])

    def predict_proba(self, X):
        raw_pred_probs = self.mdl.predict_proba(X)[:,1:]
        raw_pred_odds = np.log(raw_pred_probs/(1 - raw_pred_probs))
        return self.calib_mdl.predict_proba(raw_pred_odds)

    def get_params(self, deep=False):
        param_dict = self.mdl.get_params(deep)
        param_dict["ml_class"] = self.ml_class
        param_dict["test_size"] = self.test_size
        return param_dict

    def set_params(self, *args, **kwargs):
        self.ml_class = kwargs["ml_class"]
        self.test_size = kwargs["test_size"]
        return self.mdl.set_params(*args, **kwargs)

class PreloadedModelDeveloper:
    """
    Preloaded model
    """
    def __init__(self):
        self.init_mdl = None
        self.refit_freq = None

    def update_fit(self, new_data: Dataset):
        return

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.log(x/(1 - x))

class ModelDeveloper:
    def __init__(
        self,
        mdl,
        train_dat,
        max_features: int = None,
        refit_freq: int = None,
        max_trains: List[int] = [0],
    ):
        self.init_mdl = mdl
        self.all_train_dat = None
        self.max_trains = max_trains

        self.max_features = max_features
        self.feature_idxs = np.arange(max_features)

        self.mdls = [
            sklearn.base.clone(mdl) for _ in self.max_trains
        ]
        self.mdl_scores = np.zeros(len(self.max_trains))
        self.refit_freq = refit_freq
        self.update_fit(train_dat)

    
    @property
    def num_models(self):
        return len(self.mdls)

    def select_features(self, x_data):
        return (
            x_data[:, : self.max_features] if self.max_features is not None else x_data
        )

    def update_fit(self, new_dat: Dataset):
        raise NotImplementedError("implement please")

    @property
    def mdl_weights(self):
        exp_scores = np.exp(self.mdl_scores)
        return exp_scores/np.sum(exp_scores)

class ModelDeveloperRisk(ModelDeveloper):
    """
    This model developer outputs the log odds for the outcome.

    This model developer maintains a pool of different ML algorithms that
    have retrain the model over different window lengths (max_trains).
    It dynamically reweights candidate ML algorithms using the exponential
    weighted forecaster where the reward function is the log likelihood.
    """
    eta = 0.5

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred_risk_tot = 0
        for mdl, mdl_weight in zip(self.mdls, self.mdl_weights):
            mdl_logit = get_safe_logit(self.select_features(x), mdl).flatten()
            mdl_risk = 1/(1 + np.exp(-mdl_logit))
            pred_risk_tot += mdl_risk * mdl_weight
        return np.log(pred_risk_tot/(1 - pred_risk_tot))

    def update_fit(self, new_dat: Dataset):
        is_init_fit = self.all_train_dat is None
        self.all_train_dat = Dataset.concatenate([self.all_train_dat, new_dat])

        if (not is_init_fit) and (self.refit_freq is None):
            # do nothing
            return False

        if is_init_fit or (new_dat is not None and self.all_train_dat.size % self.refit_freq == 0):
            logging.info("refit dataset %d", self.all_train_dat.size)
            if not is_init_fit and len(self.mdls) > 1:
                # update model scores first
                for idx, (mdl, max_train) in enumerate(zip(self.mdls, self.max_trains)):
                    test_dat = self.all_train_dat.subset(
                        start_idx=max(0, self.all_train_dat.size - self.refit_freq),
                        end_idx=self.all_train_dat.size,
                    )
                    pred_logit = get_safe_logit(self.select_features(test_dat.x), mdl)
                    pred_risk = 1/(1 + np.exp(-pred_logit))
                    log_lik = test_dat.y * np.log(pred_risk) + (1 - test_dat.y) * np.log(1 - pred_risk)
                    self.mdl_scores[idx] += log_lik.sum() * self.eta

            # update underlying models
            for mdl, max_train in zip(self.mdls, self.max_trains):
                train_subdat = self.all_train_dat.subset(
                    start_idx=max(0, self.all_train_dat.size - max_train) if max_train else 0,
                    end_idx=self.all_train_dat.size,
                )
                mdl.fit(
                    self.select_features(train_subdat.x), train_subdat.y.flatten()
                )
                # logging.info("CCC %f", mdl.C_)
                # logging.info("CCC %s", mdl.Cs_)
                # logging.info(
                # print("mdl developer retrain %s %s", mdl.base_estimator.coef_, mdl.base_estimator.intercept_)
                # )
            if self.num_models > 1:
                logging.info("mdl developer weights %s", self.mdl_weights)
            return True
        else:
            print("no refitting yet")
            return False

class ModelDeveloperClassify(ModelDeveloper):
    """
    Model developer outputs outputs binary labels.
    """
    threshold = 0.7

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred_risk_tot = 0
        for mdl, mdl_weight in zip(self.mdls, self.mdl_weights):
            pred_prob = mdl.predict_proba(self.select_features(x))[:,1].flatten()
            pred_risk_tot += pred_prob * mdl_weight
        return (pred_risk_tot > self.threshold).astype(int)

    def update_fit(self, new_dat: Dataset):
        is_init_fit = self.all_train_dat is None
        self.all_train_dat = Dataset.concatenate([self.all_train_dat, new_dat])

        if not is_init_fit:
            if self.refit_freq is None:
                # do nothing
                return False
            else:
                raise NotImplementedError("not implemented yet")
        else:
            # update underlying models
            for mdl, max_train in zip(self.mdls, self.max_trains):
                train_subdat = self.all_train_dat.subset(
                    start_idx=max(0, self.all_train_dat.size - max_train) if max_train else 0,
                    end_idx=self.all_train_dat.size,
                )
                mdl.fit(
                    self.select_features(train_subdat.x), train_subdat.y.flatten()
                )
                # logging.info("CCC %f", mdl.C_)
                # logging.info("CCC %s", mdl.Cs_)
                # logging.info(
                # print("mdl developer retrain %s %s", mdl.base_estimator.coef_, mdl.base_estimator.intercept_)
                # )
            if self.num_models > 1:
                logging.info("mdl developer weights %s", self.mdl_weights)
            return True