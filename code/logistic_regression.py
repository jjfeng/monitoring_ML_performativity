"""
Helper functions with logistic regression
Mostly to get the standard error matrix
"""

import numpy as np


def get_logistic_hessian(x, y, theta):
    """hessian of the log likelihood in logistic regression

    Args:
        x (_type_): _description_
        y (_type_): ignored
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    intercept_x = np.hstack([x, np.ones((x.shape[0], 1))])

    logit_preds = np.array(np.matmul(intercept_x, theta), dtype=float)
    p_hat = (1 / (1 + np.exp(-logit_preds))).flatten()
    # variance_hat_mat = np.diag(p_hat * (1 - p_hat))
    variances = p_hat * (1 - p_hat)
    hessian = -np.sum(
        [
            intercept_x[i : i + 1].T @ intercept_x[i : i + 1] * variances[i]
            for i in range(x.shape[0])
        ],
        axis=0,
    )
    return hessian


def get_logistic_cov(x, y, theta):
    """
    @return inverse hessian, empirical fishers matrix
    """
    hessian = get_logistic_hessian(x, y, theta)
    cov_est = np.linalg.inv(-hessian)
    return cov_est


def get_logistic_gradient(x, y, theta):
    """get the gradient for the log likelihood in logistic regression

    Args:
        x (_type_): _description_
        y (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    intercept_x = np.hstack([x, np.ones((x.shape[0], 1))])
    logit_preds = np.array(np.matmul(intercept_x, theta), dtype=float)
    p_hat = (1 / (1 + np.exp(-logit_preds))).flatten()

    grad = np.multiply(intercept_x, (y.flatten() - p_hat).reshape((-1, 1)))
    return grad
