import logging

from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay

from dataset import Dataset
from recalib_monitor import RecalibMonitor
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

class CusumIPW(RecalibMonitor):
    name = "ipw_cusum"

    def __init__(
        self,
        data_gen: DataGenerator,
        alpha: float,
        confounder_start_idx: int = 0,
        confounder_end_idx: int = 0,
        batch_size: int = 1,
        num_particles: int = 2000,
        halt_when_alarm: bool = True,
    ):
        self.min_post_mtr_size = 0
        self.particles = np.zeros(num_particles)
        self.max_num_particles = num_particles
        self.prior_params = None
        self.oracle_params = None
        self.halt_when_alarm = halt_when_alarm
        logging.info("max num particles %d", num_particles)
        super().__init__(
            data_gen,
            alpha,
            confounder_start_idx,
            confounder_end_idx,
            batch_size,
            logit_powers=1
        )
    
    @property
    def num_particles(self):
        return self.particles.size

    def compute_oracle_control_lim(
        self,
        new_data: Dataset,
        pred_logit: np.ndarray,
        ipw: np.ndarray,
        t: int,
        alpha_spend: float,
    ):
        # Generate new particles using oracle model
        pred_prob = logit_to_prob(pred_logit)
        # new_mtr_x = self._create_monitor_features(pred_logit, new_data.x)
        # Just use the oracle outcome model
        new_prob_ys = self.data_gen.get_conditional_risk(new_data.x, self.data_gen.pre_beta, curr_time=-1)
        # new_prob_ys = self.oracle_outcome_mdl.predict_proba(new_mtr_x)[:,1]
        sim_ys = np.random.binomial(
            n=1, p=new_prob_ys, size=(self.num_particles, new_data.size)
        )
        
        # update particles, using oracle ipw
        ipw_briers = np.mean(np.power(sim_ys - pred_prob, 2) * ipw, axis=1)
        print("ipw_briers", ipw_briers.max(), ipw_briers.mean(), ipw_briers.min())
        self.particles = np.maximum(0, self.particles + ipw_briers - self.brier_thres)
        
        # calculate control limit
        control_lim = np.quantile(self.particles,
            q=self.max_num_particles * (1 - alpha_spend)/self.num_particles
        )

        # Determine which particles to keep
        keep_mask = self.particles <= control_lim
        self.particles = self.particles[keep_mask]
        
        print("num particles", self.num_particles, alpha_spend)

        return control_lim

    def monitor(self, max_time: int):
        hist_logger = MonitoringHistory(self.name)
        chart_stat = 0
        self.alpha_spent = 0
        max_time = min(max_time, self.data_gen.max_time - self.batch_size + 1)
        for t in range(0, max_time, self.batch_size):
            print("TIMTE", t, max_time, self.batch_size)
            # Grab data from this batch
            new_data = self.data_gen.generate_data(n=self.batch_size, curr_time=t)
            pred_logit = self.data_gen.mdl_dev.predict(new_data.x)
            pred_prob = logit_to_prob(pred_logit)

            # Calculate chart statistic
            residual = new_data.y.flatten() - pred_prob
            oracle_propensities_a0 = 1 - self.data_gen.clinician.get_propensities(new_data.x, self.data_gen.mdl_dev, curr_time=t)
            ipw = 1/oracle_propensities_a0
            logging.info("ipw max %f mean %f min %f", ipw.max(), ipw.mean(), ipw.min())
            ipw_brier = np.mean(ipw * np.power(residual, 2))
            print("oracle_propensities", oracle_propensities_a0)
            print("residual", residual)

            # Calculate chart statistic
            chart_stat = max(0, chart_stat + ipw_brier - self.brier_thres)

            # Compute control limit
            control_lim = self.compute_oracle_control_lim(
                new_data,
                pred_logit=pred_logit,
                ipw=ipw,
                t=t,
                alpha_spend=(t + self.batch_size) * self.alpha / max_time
            )

            hist_logger.update(
                [chart_stat],
                chart_stat,
                [-np.inf, control_lim],
                batch_size=self.batch_size,
                num_missing=0,
                time_idx=t,
                candidate_changepoint=None
            )

            # Check for alarm
            print("hist_logger.has_alert", hist_logger.has_alert)
            if hist_logger.has_alert:
                logging.info("ALERT FOR SHIFT: time %d", t)
                if self.halt_when_alarm:
                    break
            
            self.data_gen.mdl_dev.update_fit(new_data)

        return hist_logger
    
    def calibrate(self, n_calib: int, seed:int =-1):
        calib_data = self.data_gen.generate_data(n_calib, curr_time=seed)
        pred_logit = self.data_gen.mdl_dev.predict(calib_data.x)
        pred_prob = logit_to_prob(pred_logit)

        oracle_propensities_a0 = 1 - self.data_gen.clinician.get_propensities(calib_data.x, self.data_gen.mdl_dev, curr_time=seed)
        ipw = 1/oracle_propensities_a0
        self.brier_thres = np.mean(np.power(calib_data.y.flatten() - pred_prob, 2) * ipw)
        logging.info("self.brier_thres %f", self.brier_thres)
