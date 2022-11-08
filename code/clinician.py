"""
clinician for treatment propensities
"""
import numpy as np

from model_developers import ModelDeveloper
from common import get_pred_logit

class Clinician:
    """
    Clinician whose propensities shift on the logit scale
    """
    def __init__(
        self,
        pre_propensity_beta: np.ndarray,
        shift_propensity_beta: np.ndarray,
        shift_scale: str,
        propensity_shift_time: int = np.inf,
    ):
        self.pre_propensity_beta = pre_propensity_beta
        self.shift_propensity_beta = shift_propensity_beta
        self.shift_scale = shift_scale
        self.propensity_shift_time = propensity_shift_time

    @property
    def num_x_vars(self):
        return self.pre_propensity_beta.size - 2

    def create_propensity_inputs(self, x, mdl_dev: ModelDeveloper):
        mdl_x = mdl_dev.select_features(x)
        ml_logit = mdl_dev.predict(mdl_x).reshape((-1,1))
        if self.num_x_vars >= 1:
            prop_input_var = np.hstack([ml_logit, x[:, : self.num_x_vars]])
        else:
            prop_input_var = ml_logit
        return prop_input_var

    def get_propensities(self, x: np.ndarray, mdl_dev: ModelDeveloper, curr_time: int):
        assert self.shift_scale == "logit"
        prop_input_var = self.create_propensity_inputs(x, mdl_dev)
        prop_input_aug = np.hstack(
            [prop_input_var, np.ones((prop_input_var.shape[0], 1))]
        )
        prop_beta = (
            self.pre_propensity_beta
            if curr_time <= self.propensity_shift_time
            else (self.pre_propensity_beta + self.shift_propensity_beta)
        )
        logit_a = np.matmul(
            prop_input_aug,
            prop_beta,
        )
        prob_a = 1 / (1 + np.exp(-logit_a))
        return prob_a.flatten()

    def assign_treat(self, x: np.ndarray, mdl_dev: ModelDeveloper, curr_time: int):
        prob_a = self.get_propensities(x, mdl_dev, curr_time)
        return np.random.binomial(n=1, p=prob_a).reshape((-1, 1))

class ClinicianMultSurv(Clinician):
    """
    Clinician whose propensities are determined by the multiplicative model
    """
    def __init__(
        self,
        pre_propensity_beta_meas: np.ndarray,
        pre_propensity_beta_unmeas: np.ndarray,
        shift_propensity_beta_meas: np.ndarray,
        shift_propensity_beta_unmeas: np.ndarray,
        shift_scale: str,
        propensity_shift_time: int = np.inf,
    ):
        """
        Args:
            pre_propensity_beta_meas (np.ndarray): _description_
            pre_propensity_beta_unmeas (np.ndarray): _description_
            shift_propensity_beta_meas (np.ndarray): _description_
            shift_scale (str): _description_
            propensity_shift_time (int, optional): _description_. Defaults to np.inf.
        """
        self.pre_propensity_beta_meas = pre_propensity_beta_meas
        self.pre_propensity_beta_unmeas = pre_propensity_beta_unmeas
        assert pre_propensity_beta_meas.shape == pre_propensity_beta_unmeas.shape
        self.shift_propensity_beta_meas = shift_propensity_beta_meas
        self.shift_propensity_beta_unmeas = shift_propensity_beta_unmeas
        self.shift_scale = shift_scale
        self.propensity_shift_time = propensity_shift_time

    @property
    def num_x_vars(self):
        return self.pre_propensity_beta_meas.size - 2

    def assign_treat(self, x: np.ndarray, mdl_dev: ModelDeveloper, curr_time: int):
        assert self.shift_scale == "log_risk"
        prop_input_var = self.create_propensity_inputs(x, mdl_dev)
        prop_input_aug = np.hstack(
            [prop_input_var, np.ones((prop_input_var.shape[0], 1))]
        )

        is_pre_prop_shift = curr_time <= self.propensity_shift_time

        logit_a_unmeas = prop_input_aug @ (
            self.pre_propensity_beta_unmeas if is_pre_prop_shift else (self.pre_propensity_beta_unmeas + self.shift_propensity_beta_unmeas))
        prob_a_unmeas = 1 / (1 + np.exp(-logit_a_unmeas))
        a_unmeas = np.random.binomial(n=1, p=prob_a_unmeas).reshape((-1, 1))

        logit_a_meas = prop_input_aug @ (self.pre_propensity_beta_meas if is_pre_prop_shift else (self.pre_propensity_beta_meas + self.shift_propensity_beta_meas))
        prob_a_meas = 1 / (1 + np.exp(-logit_a_meas))
        a_meas = np.random.binomial(n=1, p=prob_a_meas).reshape((-1, 1))

        return np.maximum(a_meas, a_unmeas).flatten()