import logging

import numpy as np

from dataset import Dataset
from generate_clinician import Clinician
from model_developers import ModelDeveloper, PreloadedModelDeveloper


class DataGenerator:
    max_time = np.inf

    def __init__(
        self,
        pre_beta: np.ndarray,
        shift_beta: np.ndarray = None,
        shift_time: int = np.inf,
        clinician: Clinician = None,
        train_sampler: Clinician = None,
        train_data_rate_monitor: int = 1,
        train_data_rate_calib: int = 0,
        mdl_dev: ModelDeveloper = None,
        family: str = "bernoulli",
        scale: float = 0,
        shift_scale: str = "logit",
        shift_time_factor: float = 1,
    ):
        """
        @param family: specify outcome is bernoulli vs gaussian
        @param scale: only used for gaussian family
        """
        self.pre_beta = pre_beta
        self.shift_beta = shift_beta
        self.shift_time = shift_time
        self.clinician = clinician
        self.train_sampler = train_sampler
        self.train_data_rate_monitor = train_data_rate_monitor
        self.train_data_rate_calib = train_data_rate_calib
        self.mdl_dev = mdl_dev
        self.family = family
        self.scale = scale
        self.seed = None
        self.shift_scale = shift_scale
        self.shift_time_factor = shift_time_factor
        assert family == "bernoulli"

    def set_seed(self, seed: int):
        np.random.seed(seed)
        self.seed_offset = np.random.randint(0, 1000000)

    @property
    def num_p(self):
        return self.pre_beta.size - 1

    def generate_xdata(
        self, n: int,
    ) -> np.ndarray:
        """
        @return x data
        """
        x = (np.random.rand(n, self.num_p) - 0.5) * 2
        return x

    def generate_soc_only_xdata(
        self, n: int, curr_time: int, clinician: Clinician = None
    ) -> np.ndarray:
        """
        @return x data where a=0
        """
        x = self.generate_xdata(n)
        if clinician is not None:
            a = clinician.assign_treat(
                x, self.mdl_dev, curr_time=curr_time
            ).flatten()
        else:
            a = np.zeros(x.shape[0])
        x = x[a == 0]
        return x

    def get_conditional_risk(
        self, x: np.ndarray, pre_beta: np.ndarray, curr_time: int
    ) -> np.ndarray:
        """Generates the conditional risk for Y(0)

        Args:
            x (np.ndarray): _description_
            pre_beta (np.ndarray): _description_
            curr_time (int): _description_

        Returns:
            np.ndarray: _description_
        """
        no_shift = curr_time <= self.shift_time
        shift_beta = None if no_shift else (
                min(1, self.shift_time_factor * (curr_time - self.shift_time))
                * self.shift_beta
        )

        intercept_x = np.hstack([x, np.ones((x.shape[0], 1))])
        assert pre_beta.size == (self.num_p + 1)
        assert self.family == "bernoulli"
        if self.shift_scale == "logit":
            beta = pre_beta if no_shift else (pre_beta + shift_beta)
            logit_y = intercept_x @ beta
            prob_y = 1 / (1 + np.exp(-logit_y)).flatten()
        else:
            logit_y = intercept_x @ pre_beta
            pre_prob_y = 1 / (1 + np.exp(-logit_y))
            if no_shift:
                prob_y = pre_prob_y.flatten()
            else:
                shift_prob = intercept_x @ shift_beta
                raw_prob_y = pre_prob_y + shift_prob
                logging.info(
                    "response trimmed probabilities %f",
                    np.mean((raw_prob_y < 0) + (raw_prob_y > 1)),
                )
                print(
                    "response trimmed SMALL probabilities",
                    np.mean((raw_prob_y < 0)),
                )
                print(
                    "response trimmed BIG probabilities",
                    np.mean((raw_prob_y > 1)),
                )
                prob_y = np.minimum(1, np.maximum(0, raw_prob_y.flatten()))
        return prob_y

    def generate_y_given_x(
        self, x: np.ndarray, pre_beta: np.ndarray, curr_time: int
    ) -> np.ndarray:
        assert pre_beta.size == (self.num_p + 1)
        if self.family == "bernoulli":
            prob_y = self.get_conditional_risk(x, pre_beta, curr_time)
            y = np.random.binomial(n=1, p=prob_y).reshape((-1, 1))
        else:
            raise NotImplementedError("bad family")
        return y

    def generate_data(self, n: int, curr_time: int, factor: int = 6) -> Dataset:
        """
        Generates SOC only data
        """
        for factor_i in factor * np.arange(1, 10):
            max_n = factor_i * n
            x_big = self.generate_soc_only_xdata(max_n, curr_time=curr_time, clinician=self.clinician)
            if x_big.shape[0] >= n:
                break
        assert x_big.shape[0] >= n
        rand_choices = np.random.choice(x_big.shape[0], size=n, replace=False)
        x = x_big[rand_choices]
        a = np.zeros(n)
        y = self.generate_y_given_x(
            x,
            pre_beta=self.pre_beta,
            curr_time=curr_time,
        )
        return Dataset(x, a, y)

    def generate_training_data(self, n: int, curr_time: int, factor: int = 6) -> Dataset:
        """
        Generates data for model retraining
        """
        n_train = n * (self.train_data_rate_monitor if curr_time >= 0 else self.train_data_rate_calib)
        if n_train == 0:
            return None

        for factor_i in factor * np.arange(1, 10):
            max_n = factor_i * n_train
            x_big = self.generate_soc_only_xdata(max_n, curr_time=curr_time, clinician=self.train_sampler)
            if x_big.shape[0] >= n_train:
                break
        assert x_big.shape[0] >= n_train
        rand_choices = np.random.choice(x_big.shape[0], size=n_train, replace=False)
        x = x_big[rand_choices]
        a = np.zeros(n_train)
        y = self.generate_y_given_x(
            x,
            pre_beta=self.pre_beta,
            curr_time=curr_time,
        )
        return Dataset(x, a, y)

    def generate_oracle_data(self, n: int, curr_time: int) -> Dataset:
        x = self.generate_xdata(n)
        if self.clinician is not None:
            a = self.clinician.assign_treat(
                x, self.mdl_dev, curr_time=curr_time
            ).flatten()
        else:
            a = np.zeros(n)

        true_risk = self.get_conditional_risk(
            x,
            pre_beta=self.pre_beta,
            curr_time=curr_time,
        )
        return Dataset(x, a, true_risk)

class PreloadedDataGenerator(DataGenerator):
    def __init__(
        self,
        calib_data: Dataset,
        monitor_data: Dataset,
        family: str = "bernoulli",
    ):
        """
        @param family: specify outcome is bernoulli vs gaussian
        @param scale: only used for gaussian family
        """
        self.calib_data = calib_data
        self.monitor_data = monitor_data
        self.clinician = None
        self.mdl_dev = PreloadedModelDeveloper()
        self.family = family
        self.seed = 1
        assert family == "bernoulli"
    
    @property
    def max_time(self):
        return self.monitor_data.size

    def generate_data(self, n: int, curr_time: int) -> Dataset:
        if curr_time < 0:
            return self.calib_data
        
        assert n == 1
        return self.monitor_data.subset(curr_time, curr_time + 1)