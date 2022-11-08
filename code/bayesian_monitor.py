import logging
import copy

import numpy as np
import scipy.special
import pandas as pd
from sklearn.linear_model import LogisticRegression

import cmdstanpy as cstan

from dataset import Dataset
from logistic_regression import get_logistic_cov
from monitoring_history import MonitoringHistory
from data_generator import DataGenerator
from recalib_monitor import RecalibMonitor
from common import EPSILON

class BayesianMonitor(RecalibMonitor):
    """
    Implements Bayesian changepoint monitoring.
    Performs posterior inference using Stan.
    """
    # This is the number of columns returned by Stan that
    # just contain metadata.
    num_meta_cols = 7

    def __init__(
        self,
        data_gen: DataGenerator,
        alpha: float,
        prior_shift_factor: float,
        tmp_data_path: str,
        confounder_start_idx: int = 0,
        confounder_end_idx: int = 0,
        shift_scale: str='logit',
        batch_size: int = 1,
        chains: int = 4,  # These are defauls in CmdStan
        iter_warmup: int = 1000,  # These are defauls in CmdStan
        iter_sampling: int = 1000,  # These are defauls in CmdStan
        halt_when_alarm: bool = True,
    ):
        self.prior_shift_factor = prior_shift_factor
        self.tmp_data_path = tmp_data_path
        self.chains = chains
        self.iter_warmup = iter_warmup
        self.iter_sampling = iter_sampling
        self.sampling_init_vars = {}
        self.name = "change_%s" % shift_scale
        self.halt_when_alarm = halt_when_alarm
        if shift_scale == "logit":
            self.stan_file = "changepoint_sr.stan"
        elif shift_scale == "risk":
            self.stan_file = "changepoint_risk_sr.stan"
        else:
            raise NotImplementedError("cannot detect shifts on this scale")

        super().__init__(
            data_gen,
            alpha,
            confounder_start_idx,
            confounder_end_idx,
            batch_size,
            shift_scale=shift_scale,
        )

    def calibrate_logistic_mdl(self, n_calib: int):
        """
        Need to do special prior for logistic regression because of collinearity
        between the confounders and the predicted logit

        Sets the prior covariance matrix to the values for the standard error matrix for the
        predicted logit and the intercept, but fills in a scaled identity matrix for all the
        other confounders (scales by the variance of the estimated recalibration slope)
        """
        calib_data = self.data_gen.generate_data(n_calib, curr_time=-1)
        pred_logit = self.mdl_developer.predict(calib_data.x)
        recalib_x = self._create_monitor_features(pred_logit, calib_data.x)

        # Fit a recalibration model on the predicted logit, but no confounders
        sub_recalib_x = recalib_x[:, : self.num_logit_powers]
        lr = LogisticRegression(penalty="none")
        lr.fit(sub_recalib_x, calib_data.y)

        # Set the coefficients for the confounders to zero
        self.prechange_mean = np.concatenate(
            [lr.coef_.ravel(), [0] * self.num_confounders, lr.intercept_]
        )
        self.shift_mean = np.zeros(self.prechange_mean.shape)
        lr_params = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        logistic_se = get_logistic_cov(
            sub_recalib_x, calib_data.y, lr_params.reshape((-1, 1))
        )

        # Create the prior matrix for the pre-cahnge recalibration parameters
        self.prechange_cov = np.eye(self.prechange_mean.size) * logistic_se[0, 0]
        self.prechange_cov[
            : self.num_logit_powers, : self.num_logit_powers
        ] = logistic_se[: self.num_logit_powers, : self.num_logit_powers]
        self.prechange_cov[: self.num_logit_powers, -1] = logistic_se[
            : self.num_logit_powers, -1
        ]
        self.prechange_cov[-1, : self.num_logit_powers] = logistic_se[
            -1, : self.num_logit_powers
        ]
        self.prechange_cov[-1, -1] = logistic_se[-1, -1]

        # Create the prior matrix for the parameter shifts
        min_diag = np.min(np.diag(self.prechange_cov))
        if self.shift_scale == "logit":
            shift_cov_scale = np.power(np.abs(self.prior_shift_factor * self.prechange_mean).max(), 2)
            self.shift_cov = np.diag(np.diag(self.prechange_cov)/min_diag) * max(min_diag * 10, shift_cov_scale)
        else:
            self.shift_cov = np.diag(np.diag(self.prechange_cov)/min_diag) * self.prior_shift_factor
        logging.info("shift mean %s", self.shift_mean)
        logging.info("PRECHANGE MEAN %s", self.prechange_mean)
        logging.info("PRECHANGE COV %s", self.prechange_cov)
        logging.info("SHIFT COV %s", self.shift_cov)
        print(self.prechange_mean)
        print(self.prechange_cov)

    def calibrate_nonlinear_mdl(self, n_calib: int):
        calib_data = self.data_gen.generate_data(n_calib, curr_time=-1)
        pred_logit = self.mdl_developer.predict(calib_data.x)
        recalib_x = self._create_monitor_features(pred_logit, calib_data.x)
        lr = LogisticRegression(penalty="none")
        lr.fit(recalib_x, calib_data.y)
        self.prechange_mean = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        logistic_se = get_logistic_cov(
            recalib_x, calib_data.y, self.prechange_mean.reshape((-1, 1))
        )
        self.prechange_cov = logistic_se
        self.shift_mean = np.zeros(self.prechange_mean.shape)

        # Create the prior matrix for the parameter shifts
        min_diag = np.min(np.diag(self.prechange_cov))
        if self.shift_scale == "logit":
            shift_cov_scale = np.power(np.abs(self.prior_shift_factor * self.prechange_mean).max(), 2)
            self.shift_cov = np.diag(np.diag(self.prechange_cov)/min_diag) * max(min_diag * 10, shift_cov_scale)
        else:
            self.shift_cov = np.diag(np.diag(self.prechange_cov)/min_diag) * self.prior_shift_factor

        logging.info("PRECHANGE MEAN %s", self.prechange_mean)
        logging.info("PRECHANGE COV %s", self.prechange_cov)
        print(self.prechange_mean)
        print(self.prechange_cov)

    def calibrate(self, n_calib: int):
        if self.mdl_developer.init_mdl is None or self.mdl_developer.init_mdl.__class__.__name__ == "LogisticRegression":
            self.calibrate_logistic_mdl(n_calib)
        else:
            self.calibrate_nonlinear_mdl(n_calib)

    def set_prior(self, prior_dict: dict):
        self.prechange_mean = np.array(prior_dict["prechange_mean"])
        self.prechange_cov = np.eye(self.prechange_mean.size) * 1e-10
        self.shift_mean = np.array(prior_dict["shift_mean"])
        self.shift_cov = self.prechange_cov
        self.sampling_init_vars = {
            "theta0": self.prechange_mean.tolist(),
            "theta1": (self.prechange_mean + self.shift_mean).tolist(),
        }
        self.chains = 1
        self.iter_sampling = 1
        self.iter_warmup = 1

    def monitor(self, max_time: int, no_change_prob: float = 0.5):
        mtr_history = MonitoringHistory(self.name)
        data = self.data_gen.generate_data(n=1, curr_time=0)
        pred_logit = self.mdl_developer.predict(data.x)

        # this is the prior from Shiryaev-Roberts, but over a finite time period.
        # we set p in the geometric distribution to p = 1/max_time
        # We set the probability to absolutely no change to the alarm rate
        all_change_probs = 1/max_time * np.power(1 - 1/max_time, np.arange(1, max_time + 1))
        all_change_probs[:-1] = all_change_probs[:-1] * (1 - no_change_prob)/np.sum(all_change_probs[:-1])
        all_change_probs[-1] = no_change_prob 

        model = cstan.CmdStanModel(stan_file=self.stan_file)
        chart_stat = 0
        max_time = min(self.data_gen.max_time - self.batch_size, max_time)
        print("mAX TIME", max_time)
        for t in range(2, max_time):
            logging.info("TIME %d", t)
            new_data = self.data_gen.generate_data(n=1, curr_time=t)
            data = Dataset.concatenate([data, new_data])
            new_pred_logit = self.mdl_developer.predict(new_data.x)
            pred_logit = np.concatenate([pred_logit, new_pred_logit])
            monitor_x = self._create_monitor_features(pred_logit, data.x)
            delta_x = self._create_delta_features(pred_logit, data.x)

            change_probs = all_change_probs[:t].copy()
            change_probs[-1] = 1 - np.sum(change_probs[:-1])

            stan_dict = {
                "tot_T": max_time,
                "T": t,
                "p": monitor_x.shape[1],
                "delta_p": delta_x.shape[1],
                "x": monitor_x,
                "delta_x": delta_x,
                "y": data.y.flatten(),
                "prechange_mean": self.prechange_mean,
                "prechange_cov": self.prechange_cov,
                "shift_mean": self.shift_mean,
                "shift_cov": self.shift_cov,
                "prior_change_probs": change_probs
            }
            if (t - 2) % self.batch_size == 0:
                print("TIME", t)
                # Update chart statistic because batch has been completely collected
                cstan.write_stan_json(self.tmp_data_path, stan_dict)
                posterior_fit = model.sample(
                    data=self.tmp_data_path,
                    chains=self.chains,
                    iter_warmup=self.iter_warmup,
                    iter_sampling=self.iter_sampling,
                    inits=self.sampling_init_vars,
                )
                posterior_df = posterior_fit.draws_pd()
                posterior_meta_df = posterior_df.iloc[:, : self.num_meta_cols]
                posterior_logistic_df = posterior_df.iloc[
                    :,
                    self.num_meta_cols : (
                        self.num_meta_cols
                        + 2 * (1 + self.num_logit_powers + self.num_confounders)
                    ),
                ]
                assert not np.any(np.isnan(posterior_df.iloc[:, -10:]).sum())
                # TODO: add check that the posterior has converged
                theta_post_means = posterior_logistic_df.mean()
                print("POSTERIOR MEAN", posterior_logistic_df.mean())
                logging.info("logistic posterior %s", posterior_logistic_df.mean())

                if self.iter_sampling != 1:
                    self.sampling_init_vars = {
                        "theta0": theta_post_means[: self.prechange_mean.size].tolist(),
                        "theta1": theta_post_means[self.prechange_mean.size :].tolist(),
                    }
                    # self.iter_warmup = 100
                chart_stat, tau_max = self.get_posterior_prob(
                    curr_t=t, posterior_df=posterior_df
                )
                print("TAU MAX", tau_max)

            mtr_history.update(
                [chart_stat],
                chart_stat,
                control_lim=[-np.inf, 1 - self.alpha],
                batch_size=1,
                num_missing=0,
                time_idx=t,
                candidate_changepoint=tau_max,
            )
            if chart_stat > (1 - self.alpha):
                logging.info("fire alarm")
                if self.halt_when_alarm:
                    break

            self.mdl_developer.update_fit(new_data)

        return mtr_history

    def get_posterior_prob(self, curr_t: int, posterior_df: pd.DataFrame) -> float:
        # Get the posterior probs for any change occuring prior to current time point
        posterior_changeprob_df = posterior_df.iloc[:, -curr_t:]
        log_denom = scipy.special.logsumexp(posterior_changeprob_df, axis=1)
        log_numer = posterior_changeprob_df.iloc[:, -1]
        cond_change_prob = 1 - np.exp(log_numer - log_denom)
        posterior_change_prob = np.mean(cond_change_prob)
        logging.info("post prob %f", posterior_change_prob)
        print("posterior prob", posterior_change_prob)
        assert not np.isnan(posterior_change_prob)

        # Get the posterior probs for the individual change times
        change_tau_prob = np.mean(np.exp(posterior_changeprob_df.iloc[:, :-1] - log_denom.reshape((-1,1))), axis=0)
        tau_max = np.argmax(change_tau_prob)

        return posterior_change_prob, tau_max