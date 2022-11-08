import logging

import numpy as np


class Dataset:
    def __init__(self, x, a, y):
        self.x = x
        self.a = a
        self.y = y
        assert a.shape[0] == y.shape[0]
        assert x.shape[0] == y.shape[0]

    @property
    def aug_x(self) -> np.ndarray:
        return np.hstack([self.x, np.ones((self.x.shape[0], 1))])

    def subset_features(self, max_features: int):
        return Dataset(
            self.x[:, :max_features],
            self.a,
            self.y,
        )

    def subset(self, start_idx, end_idx):
        return Dataset(
            self.x[start_idx:end_idx],
            self.a[start_idx:end_idx],
            self.y[start_idx:end_idx],
        )

    def subset_idxs(self, idxs):
        return Dataset(
            self.x[idxs],
            self.a[idxs],
            self.y[idxs],
        )

    @property
    def size(self):
        return self.x.shape[0]

    @staticmethod
    def concatenate(datasets):
        if sum([d is not None for d in datasets]) > 1:
            return Dataset(
                np.concatenate([d.x for d in datasets if d is not None]),
                np.concatenate([d.a for d in datasets if d is not None]),
                np.concatenate([d.y for d in datasets if d is not None]),
            )
        else:
            for d in datasets:
                if d is not None:
                    return d
