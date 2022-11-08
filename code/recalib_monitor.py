import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from data_generator import DataGenerator
from model_developers import ModelDeveloper


class RecalibMonitor:
    """
    Class for monitoring model calibration.
    Bayesian and score-based CUSUM inherit from this class
    """
    def __init__(
        self,
        data_gen: DataGenerator,
        alpha: float,
        confounder_start_idx: int = 0,
        confounder_end_idx: int = 0,
        batch_size: int = 1,
        shift_scale: str = 'logit',
        logit_powers: int = 1,
    ):
        self.data_gen = data_gen
        self.alpha = alpha
        self.confounder_start_idx = confounder_start_idx
        self.confounder_end_idx = confounder_end_idx
        self.confounder_idxs = np.arange(confounder_start_idx, confounder_end_idx)
        self.num_confounders = self.confounder_end_idx - self.confounder_start_idx
        self.num_logit_powers = logit_powers
        self.batch_size = batch_size
        self.shift_scale = shift_scale

    @property
    def clinician(self):
        return self.data_gen.clinician

    @property
    def mdl_developer(self) -> ModelDeveloper:
        return self.data_gen.mdl_dev

    def _create_monitor_features(self, pred_logit, x):
        pred_logit_features = pred_logit.reshape((-1, 1))
        assert self.num_logit_powers == 1

        if self.confounder_start_idx == self.confounder_end_idx:
            return pred_logit_features
        else:
            return np.hstack(
                [
                    pred_logit_features,
                    x[:, self.confounder_start_idx : self.confounder_end_idx],
                ]
            )

    def _create_delta_features(self, pred_logit, x):
        if self.shift_scale == "logit":
            mdl_pred_features = pred_logit.reshape((-1,1))
        else:
            pred_prob = 1 / (1 + np.exp(-pred_logit))
            mdl_pred_features = pred_prob.reshape((-1,1))
        if self.confounder_start_idx == self.confounder_end_idx:
            return mdl_pred_features
        else:
            return np.hstack(
                [
                    mdl_pred_features,
                    x[:, self.confounder_start_idx : self.confounder_end_idx],
                ]
            )


