import numpy as np


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
