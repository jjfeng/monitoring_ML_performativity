import logging
import time
import copy

import numpy as np
from sklearn.linear_model import LogisticRegression

from dataset import Dataset
from data_generator import DataGenerator
from logistic_regression import get_logistic_gradient, get_logistic_hessian
from monitoring_history import MonitoringHistory
from recalib_monitor import RecalibMonitor


class ScoreMonitor(RecalibMonitor):
    """
    Implements the score-based CUSUM procedure
    """
    name = "score"

    def __init__(
        self,
        data_gen: DataGenerator,
        norm,
        alpha: float,
        confounder_start_idx: int = 0,
        confounder_end_idx: int = 0,
        shift_scale: str = "logit",
        batch_size: int = 1,
        max_look_back: int = 0,
        num_particles: int = 2000,
        halt_when_alarm: bool = True,
    ):
        self.norm = norm
        self.min_post_mtr_size = 0
        self.num_particles = num_particles
        self.max_num_particles = num_particles
        self.prior_params = None
        self.oracle_params = None
        self.max_look_back = max_look_back
        self.shift_scale = shift_scale
        self.halt_when_alarm = halt_when_alarm
        logging.info("max num particles %d", num_particles)
        super().__init__(
            data_gen,
            alpha,
            confounder_start_idx,
            confounder_end_idx,
            batch_size,
            shift_scale=shift_scale,
            logit_powers=1
        )

    def calibrate(self, n_calib: int, penalty: str = "none", seed=-1):
        """
        This will initialize the score-based CUSUM procedure
        by analyzing n_calib observations (non-contaminated).
        """
        pred_logit = []
        calib_data = []
        mdl_hist = []
        for time_idx in range(seed - n_calib, seed, self.batch_size):
            calib_obs = self.data_gen.generate_data(n=self.batch_size, curr_time=time_idx)
            calib_data.append(calib_obs)
            pred_logit.append(self.mdl_developer.predict(calib_obs.x).flatten())

            train_obs = self.data_gen.generate_training_data(
                n= self.batch_size,
                curr_time=time_idx)
            did_refit = self.mdl_developer.update_fit(train_obs)
            if did_refit:
                print("DID REFIT")
                mdl_hist.append({
                    "time": time_idx,
                    "mdl": copy.deepcopy(self.mdl_developer)
                })
        pred_logit = np.concatenate(pred_logit)
        calib_data = Dataset.concatenate(calib_data)

        recalib_x = self._create_monitor_features(pred_logit, calib_data.x)
        delta_x = self._create_delta_features(pred_logit, calib_data.x)
        self.monitor_data = Dataset(recalib_x, delta_x, calib_data.y)
        self.monitor_start = self.monitor_data.size

        pool_lr = LogisticRegression(penalty=penalty)
        pool_lr.fit(self.monitor_data.x, self.monitor_data.y.ravel())
        pred_prob_y = pool_lr.predict_proba(self.monitor_data.x)[:, 1]
        pool_params = np.concatenate(
            [pool_lr.coef_.flatten(), pool_lr.intercept_]
        ).reshape((-1, 1))

        self.hessian_cusum = np.sum(
            [
                get_logistic_hessian(
                    self.monitor_data.x[i : i + 1],
                    self.monitor_data.y[i : i + 1],
                    pool_params,
                )
                for i in range(self.monitor_data.size)
            ],
            axis=0,
            keepdims=True,
        )
        self.inv_hessians = np.array([np.linalg.inv(self.hessian_cusum[0])])

        # calculate the score for all possible y values (both 0 and 1)
        tiled_x = np.repeat(self.monitor_data.x, repeats=2, axis=0)
        tiled_y = np.tile([0, 1], reps=self.monitor_data.size).reshape((-1, 1))
        scores_theta = get_logistic_gradient(tiled_x, tiled_y, pool_params.flatten())
        # Parametric bootstrap for the calibration dataset to get distribution of the initial chart statistic
        particle_rand_choices = 2 * np.arange(
            0, self.monitor_data.size
        ) + np.random.binomial(
            n=1, p=pred_prob_y, size=(self.num_particles, self.monitor_data.size)
        )
        particles_theta_score = np.array(
            [scores_theta[particle_rand_choices[i]] for i in range(self.num_particles)]
        )
        self.particles_theta_score_cum = particles_theta_score.sum(
            axis=1, keepdims=True
        )
        self.particles_theta_inf_func_avgs = self.inv_hessians[-1] @ np.expand_dims(
            self.particles_theta_score_cum, axis=3
        )

        # The cumulative sums of the delta scores and its gradient wrt theta are initialized at zero
        self.particles_score_delta_cusum = np.zeros(
            (self.num_particles, 1, self.monitor_data.a.shape[1] + 1)
        )
        self.particles_correction_cusum = np.zeros(
            (self.num_particles, 1, self.monitor_data.a.shape[1] + 1, 1)
        )

        # SANITY CHECK STUFF
        particle_obs = calib_data.y.flatten() + 2 * np.arange(0, self.monitor_data.size)
        assert np.all(tiled_x[particle_obs] == self.monitor_data.x)
        assert np.all(tiled_y[particle_obs].flatten() == calib_data.y.flatten())

        return pool_params, mdl_hist

    def _get_delta_theta_gradient_exp(self, mtr_x, delta_x, pool_params):
        delta_feats_aug = np.hstack([delta_x, np.ones((delta_x.shape[0], 1))])
        mtr_feats_aug = np.hstack([mtr_x, np.ones((mtr_x.shape[0], 1))])
        delta_feats_aug_vecs = delta_feats_aug.reshape(
            (delta_feats_aug.shape[0], delta_feats_aug.shape[1], 1)
        )
        mtr_feats_aug_vecsT = mtr_feats_aug.reshape(
            (mtr_feats_aug.shape[0], 1, mtr_feats_aug.shape[1])
        )
        feats_sq = np.matmul(delta_feats_aug_vecs, mtr_feats_aug_vecsT)
        recalib_logit = np.matmul(mtr_feats_aug, pool_params)
        pre_change_prob = 1 / (1 + np.exp(-recalib_logit)).reshape((-1, 1, 1))
        if self.shift_scale == "logit":
            variances = pre_change_prob * (1 - pre_change_prob)
            delta_theta_scores = -feats_sq * variances
        else:
            delta_theta_scores = (
                (1 - pre_change_prob)
                * pre_change_prob
                * feats_sq
                * (
                    -np.power(pre_change_prob, -2) * pre_change_prob
                    + np.power(1 - pre_change_prob, -2) * (1 - pre_change_prob)
                )
            )
        return delta_theta_scores


    def _get_delta_gradient(self, mtr_x, delta_x, y, pool_params):
        delta_feats_aug = np.hstack([delta_x, np.ones((delta_x.shape[0], 1))])
        mtr_feats_aug = np.hstack([mtr_x, np.ones((mtr_x.shape[0], 1))])
        recalib_logit = np.matmul(mtr_feats_aug, pool_params)
        pre_change_prob = 1 / (1 + np.exp(-recalib_logit))
        if self.shift_scale == "logit":
            residual = (y.flatten() - pre_change_prob.flatten()).reshape((-1, 1))
            delta_scores = np.multiply(
                delta_feats_aug,
                residual)
        else:
            delta_scores = 1 / pre_change_prob * delta_feats_aug * y - (
                1 / (1 - pre_change_prob)
            ) * delta_feats_aug * (1 - y)
        return delta_scores

    def get_chart_statistic(self, t: int):
        all_post_data = self.monitor_data.subset(
            self.monitor_start, self.monitor_start + t
        )
        all_delta_scores = np.array(
            [
                self._get_delta_gradient(
                    all_post_data.x[i : i + 1],
                    all_post_data.a[i : i + 1],
                    all_post_data.y[i : i + 1],
                    self.pre_params[i],
                )
                for i in range(t)
            ]
        )
        chart_st = []
        for i in range(0, t, self.batch_size):
            delta_score = np.sum(all_delta_scores[i:t], axis=0).flatten()
            chart_st.append(np.linalg.norm(delta_score, ord=self.norm))
        chart_stat = np.max(chart_st[-self.max_look_back :])
        print("CHART STAT", chart_stat)
        max_change_idx = np.argmax(chart_st)
        logging.info("max shift time %d", np.argmax(chart_st))
        print("max shift time %d", np.argmax(chart_st))
        max_delta_score = np.mean(all_delta_scores[max_change_idx * self.batch_size:t], axis=0)
        logging.info("max delta %s", max_delta_score)
        print("max delta %s", max_delta_score)
        
        return chart_stat, max_change_idx

    def compute_control_lim(
        self,
        pool_lr: LogisticRegression,
        new_mtr_data: Dataset,
        t: int,
        alpha_spend: float,
    ):
        # Generate new particle
        new_prob_ys = pool_lr.predict_proba(new_mtr_data.x)[:, 1]
        sim_ys = np.random.binomial(
            n=1, p=new_prob_ys, size=(self.num_particles, new_mtr_data.size)
        )
        particle_rand_choices = 2 * np.arange(0, new_mtr_data.size) + sim_ys

        tiled_x = np.repeat(new_mtr_data.x, repeats=2, axis=0)
        tiled_a = np.repeat(new_mtr_data.a, repeats=2, axis=0)
        tiled_y = np.tile([0, 1], reps=new_mtr_data.size).reshape((-1, 1))
        theta0 = (
            self.pre_params[-1] if self.oracle_params is None else self.oracle_params
        )
        possible_scores_theta = get_logistic_gradient(
            tiled_x, tiled_y, theta0.flatten()
        )
        sim_theta_score = np.sum(
            possible_scores_theta[particle_rand_choices], axis=1, keepdims=True
        )
        possible_scores_delta = self._get_delta_gradient(
            tiled_x, tiled_a, tiled_y, theta0
        )
        sim_score_delta = np.sum(
            possible_scores_delta[particle_rand_choices], axis=1, keepdims=True
        )
        delta_theta_grad_exp = np.sum(
            self._get_delta_theta_gradient_exp(new_mtr_data.x, new_mtr_data.a, theta0),
            axis=0,
            keepdims=True,
        )

        # Append to particles
        self.particles_theta_score_cum = self.particles_theta_score_cum + sim_theta_score
        self.particles_theta_inf_func_avgs = np.concatenate(
            [
                self.particles_theta_inf_func_avgs,
                -np.matmul(
                    self.inv_hessians[-1],
                    np.expand_dims(self.particles_theta_score_cum, axis=3),
                ),
            ],
            axis=1,
        )
        # For delta score: On axis 1, index i is cumulative sum from time i (post recalib) to current time t
        self.particles_score_delta_cusum += sim_score_delta
        self.particles_score_delta_cusum = np.concatenate(
            [self.particles_score_delta_cusum, np.zeros(sim_score_delta.shape)], axis=1
        )
        # For delta score: On axis 1, index i is cumulative sum from time i (post recalib) to current time t
        self.particles_correction_cusum += (
            delta_theta_grad_exp @ self.particles_theta_inf_func_avgs[:, -2:-1]
        )
        self.particles_correction_cusum = np.concatenate(
            [
                self.particles_correction_cusum,
                np.zeros(
                    (
                        self.particles_correction_cusum.shape[0],
                        1,
                        self.particles_correction_cusum.shape[2],
                        self.particles_correction_cusum.shape[3],
                    )
                ),
            ],
            axis=1,
        )

        if (t + self.batch_size) <= self.min_post_mtr_size:
            return 1

        # Calculate chart stat for each particle
        # Get distribution of chart stats
        delta_part = np.expand_dims(self.particles_score_delta_cusum, axis=3)
        if self.prior_params is not None:
            post_pre_diffs_raw = delta_part
        else:
            post_pre_diffs_raw = delta_part + self.particles_correction_cusum
        post_pre_diffs = np.linalg.norm(post_pre_diffs_raw[:, :, :, 0], ord=self.norm, axis=2)
        print("SIMULATE PARTICLE SHAPE", post_pre_diffs.shape)
        if self.max_look_back > 0:
            particle_chart_stats = np.max(post_pre_diffs[:, -(self.max_look_back + 1) :], axis=1)
        else:
            particle_chart_stats = np.max(post_pre_diffs, axis=1)

        if alpha_spend == 0:
            return 1

        control_lim = np.quantile(particle_chart_stats,
            q=self.max_num_particles * (1 - alpha_spend)/particle_chart_stats.shape[0]
        )

        # Determine which particles to keep
        keep_mask = particle_chart_stats <= control_lim
        self.particles_theta_score_cum = self.particles_theta_score_cum[keep_mask]
        self.particles_theta_inf_func_avgs = self.particles_theta_inf_func_avgs[
            keep_mask
        ]
        self.particles_score_delta_cusum = self.particles_score_delta_cusum[keep_mask]
        self.particles_correction_cusum = self.particles_correction_cusum[keep_mask]
        self.num_particles = np.sum(keep_mask)

        return control_lim

    def monitor(self, max_time):
        hist_logger = MonitoringHistory(self.name)
        chart_stat = 0
        control_lim = 1
        self.alpha_spent = 0
        self.pre_params = []
        pool_lr = LogisticRegression(penalty="none", warm_start=True, max_iter=2000)
        max_time = min(max_time, self.data_gen.max_time - self.batch_size + 1)
        for t in range(0, max_time, self.batch_size):
            print("TIME", t)
            logging.info("TIME %d", t)
            pool_lr.fit(self.monitor_data.x, self.monitor_data.y.ravel())
            print("POOL LR", pool_lr.coef_, pool_lr.intercept_)
            if self.prior_params is None:
                self.pre_params += [
                    np.concatenate(
                        [pool_lr.coef_.flatten(), pool_lr.intercept_]
                    ).reshape((-1, 1))
                ] * self.batch_size
            else:
                pool_lr.coef_[:] = self.prior_params.flatten()[:-1]
                pool_lr.intercept_[:] = self.prior_params.flatten()[-1:]
                self.pre_params += [self.prior_params] * self.batch_size
            if self.oracle_params is not None:
                pool_lr.coef_[:] = self.oracle_params.flatten()[:-1]
                pool_lr.intercept_[:] = self.oracle_params.flatten()[-1:]
            logging.info("pre params %s", self.pre_params[-1].flatten())

            # Grab data from this batch
            new_data = self.data_gen.generate_data(n=self.batch_size, curr_time=t)
            pred_logit = self.data_gen.mdl_dev.predict(new_data.x)
            new_recalib_x = self._create_monitor_features(pred_logit, new_data.x)
            new_delta_x = self._create_delta_features(pred_logit, new_data.x)
            new_mtr_data = Dataset(new_recalib_x, new_delta_x, new_data.y)
            self.monitor_data = Dataset.concatenate([self.monitor_data, new_mtr_data])

            # Calculate chart statistic
            check_chart = (t + self.batch_size) > self.min_post_mtr_size
            if check_chart:
                chart_stat, candidate_changepoint = self.get_chart_statistic(t=t + self.batch_size)

            new_hessian = get_logistic_hessian(
                new_recalib_x,
                new_data.y,
                (
                    self.pre_params[-1]
                    if self.oracle_params is None
                    else self.oracle_params
                ).flatten(),
            )
            self.hessian_cusum = np.concatenate(
                [
                    self.hessian_cusum,
                    self.hessian_cusum[-1:] + np.expand_dims(new_hessian, axis=0),
                ]
            )
            self.inv_hessians = np.concatenate(
                [self.inv_hessians, [np.linalg.inv(self.hessian_cusum[-1])]]
            )

            control_lim = self.compute_control_lim(
                pool_lr,
                new_mtr_data,
                t=t,
                alpha_spend=(t + self.batch_size) * self.alpha / max_time
                if check_chart
                else 0,
            )
            logging.info("chart stat %f", chart_stat)
            logging.info("control lim %f", control_lim)
            print("chart", chart_stat, "control", control_lim)

            hist_logger.update(
                [chart_stat],
                chart_stat,
                [-np.inf, control_lim],
                batch_size=self.batch_size,
                num_missing=0,
                time_idx=t,
                candidate_changepoint=candidate_changepoint,
                mdl=None if self.data_gen.mdl_dev.refit_freq is None else (
                    # track the updated model if it has been updated
                    self.data_gen.mdl_dev if (self.data_gen.mdl_dev.all_train_dat.size % self.data_gen.mdl_dev.refit_freq) == 0 else None)
            )

            # Check for alarm
            if hist_logger.has_alert:
                logging.info("ALERT FOR SHIFT: time %d", t)
                if self.halt_when_alarm:
                    break

            self.data_gen.mdl_dev.update_fit(new_data)

        return hist_logger
