import logging

import numpy as np

from data_generator import DataGenerator
from monitoring_history import MonitoringHistory
from common import logit_to_prob, get_log_lik


class Cusum:
    def __init__(self, prechange_prob: float, postchange_prob: float):
        self.prechange_prob = prechange_prob
        self.postchange_prob = postchange_prob
        self.chart_stat_hist = [0]
        self.chart_stat = 0
        self.candidate_changepoint = 0
        self.time_idx = 0

    def update(self, new_y: np.ndarray):
        pre_loglik = get_log_lik(self.prechange_prob, new_y)
        post_loglik = get_log_lik(self.postchange_prob, new_y)
        self.time_idx += 1
        if self.chart_stat + post_loglik - pre_loglik < 0:
            self.candidate_changepoint = self.time_idx
        self.chart_stat = max(0, self.chart_stat + post_loglik - pre_loglik)
        self.chart_stat_hist.append(self.chart_stat)

class CusumNaive:
    name = "cusum_naive"
    num_strata = 1
    batch_size = 1

    def __init__(
        self,
        data_gen: DataGenerator,
        alpha: float,
        max_time: int,
        detect_incr: float = 0.1
    ):
        self.data_gen = data_gen
        self.alpha = alpha
        self.max_time = max_time
        self.detect_incr = detect_incr
    
    def calibrate(self, n_calib: int, seed:int =-1):
        calib_data = self.data_gen.generate_data(n_calib, curr_time=seed)
        predictions = self.data_gen.mdl_dev.predict(calib_data.x)
        prechange_prob = np.mean(predictions != calib_data.y.flatten())
        postchange_prob = prechange_prob + self.detect_incr
        logging.info("prechange_prob %f", prechange_prob)
        logging.info("postchange_prob %f", postchange_prob)

        control_lim = self.calc_control_limit(
            prechange_prob,
            postchange_prob,
        )
        self.cusum = Cusum(
                prechange_prob=prechange_prob,
                postchange_prob=postchange_prob,
        )
        self.control_lim = control_lim

    def calc_control_limit(
        self,
        prechange_prob: float,
        postchange_prob: float,
        num_simulations: int = 500,
        step_size: float = 0.1,
    ):
        cusum_alpha = self.alpha / self.num_strata  # bonferroni correction

        def sim_chart_stat():
            y_sim = np.random.binomial(n=1, p=prechange_prob, size=self.max_time)
            chart_stat_mtr = Cusum(
                prechange_prob=prechange_prob, postchange_prob=postchange_prob
            )
            for y in y_sim:
                chart_stat_mtr.update(y)
            return chart_stat_mtr.chart_stat_hist

        chart_stat_hists = np.array([sim_chart_stat() for i in range(num_simulations)])
        for control_lim in np.arange(0, 10, step=step_size):
            alarm_rate = np.mean(np.any(chart_stat_hists > control_lim, axis=1))
            if alarm_rate < cusum_alpha:
                print("alarm rate", alarm_rate, control_lim)
                return control_lim + step_size
        raise ValueError("could not find control limit")

    def monitor(self):
        hist_logger = MonitoringHistory(self.name)
        for t in range(self.max_time):
            # Grab data from this batch
            new_data = self.data_gen.generate_data(n=self.batch_size, curr_time=t)

            pred_class = self.data_gen.mdl_dev.predict(new_data.x)
            self.cusum.update(int(pred_class != new_data.y.flatten()))

            hist_logger.update(
                [self.cusum.chart_stat],
                self.cusum.chart_stat,
                [-np.inf, self.control_lim],
                batch_size=self.batch_size,
                num_missing=0,
                time_idx=t,
                candidate_changepoint=self.cusum.candidate_changepoint,
            )

            # Check for alarm
            if hist_logger.has_alert:
                logging.info("ALERT FOR SHIFT: time %d", t)
                break

            self.data_gen.mdl_dev.update_fit(new_data)

        return hist_logger