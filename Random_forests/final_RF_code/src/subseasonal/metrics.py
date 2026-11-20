import numpy as np

def weighted_rmse_step(y_true, y_pred, z_obs):
    z_abs = np.abs(z_obs)
    weights = np.where(z_abs < 1, 1.0,
               np.where(z_abs < 2, 2.0, 3.0))
    return np.sqrt(np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights))


def weighted_rmse_smooth(y_true, y_pred, z_obs, alpha=1.0):
    z_abs = np.abs(z_obs)
    weights = 1.0 + alpha * np.maximum(0.0, z_abs - 1.0)
    return np.sqrt(np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights))