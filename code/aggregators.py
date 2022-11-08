import logging

import numpy as np


class AggregatorMax:
    compare_max = True

    def __init__(self, xi_dim: int, max_looks: int = None):
        self.xi_dim = xi_dim
        self.max_looks = max_looks
        assert max_looks > 0

    @property
    def name(self):
        return "aggmax_%d" % self.max_looks

    def agg_particles(self, xi_cumsums):
        assert xi_cumsums.shape[1] == self.xi_dim

        if self.xi_dim > 1:
            norm_sq_cumsums = np.sum(
                np.power(xi_cumsums[:, :, -self.max_looks :], 2), axis=1
            )
            if norm_sq_cumsums.shape[0] == 1:
                logging.info("arg max %s", np.argmax(norm_sq_cumsums, axis=1)[0])
            max_val = np.amax(norm_sq_cumsums, axis=1)
            max_idx = np.argmax(norm_sq_cumsums, axis=1)
        else:
            if xi_cumsums.shape[0] == 1:
                print("xi cumsum", xi_cumsums[:, 0, -self.max_looks :])
                logging.info(
                    "arg max %s",
                    np.argmax(xi_cumsums[:, 0, -self.max_looks :], axis=1)[0],
                )
            max_val = np.amax(xi_cumsums[:, 0, -self.max_looks :], axis=1)
            max_idx = np.argmax(xi_cumsums[:, 0, -self.max_looks :], axis=1)
        return max_val, max_idx


class AggregatorWindowSum:
    compare_max = True

    def __init__(self, xi_dim: int, window_size=None):
        self.xi_dim = xi_dim
        self.window_size = window_size
        self.max_looks = window_size

    @property
    def name(self):
        return "windowsum_%d_%d" % (self.window_size, self.max_looks)

    def agg_particles(self, xi_cumsums):
        assert xi_cumsums.shape[1] == self.xi_dim
        idx = max(0, xi_cumsums.shape[2] - self.window_size)
        if self.xi_dim > 1:
            max_val = np.sum(np.power(xi_cumsums[:, :, idx], 2), axis=1)
        else:
            max_val = xi_cumsums[:, 0, idx]
        return max_val, np.zeros(xi_cumsums.shape[0], dtype=int)


class AggregatorCumSum:
    compare_max = False

    def __init__(self, xi_dim: int, max_looks: int = 0):
        self.xi_dim = xi_dim
        self.max_looks = max_looks

    @property
    def name(self):
        return "cumsum_%d_%d" % self.max_looks

    def agg_particles(self, xi_cumsums):
        assert xi_cumsums.shape[1] == self.xi_dim
        if self.xi_dim > 1:
            max_val = np.sum(np.power(xi_cumsums[:, :, 0], 2), axis=1)
        else:
            max_val = xi_cumsums[:, 0, 0]
        return max_val, np.zeros(xi_cumsums.shape[0], dtype=int)


def make_aggregator(agg_str, xi_dim: int, max_looks: int = None):
    if agg_str == "cum":
        return AggregatorCumSum(xi_dim, max_looks)
    if agg_str == "window_cum":
        return AggregatorWindowSum(xi_dim, window_size=max_looks)
    else:
        return AggregatorMax(xi_dim, max_looks)
