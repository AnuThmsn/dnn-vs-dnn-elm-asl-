"""
calculate_metrics_v2.py
Metrics calculation module for V2:
Computes MSE, RMSE, MAE, Bias, Lin's CCC, and percentage gains.
"""

import numpy as np

def compute_lins_ccc(y_true, y_pred):
    """
    Computes Lin's Concordance Correlation Coefficient (CCC):
      CCC = (2 * s_xy) / (s_x^2 + s_y^2 + (x_bar - y_bar)^2)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)

    var_t = np.var(y_true, ddof=1)
    var_p = np.var(y_pred, ddof=1)

    cov_tp = np.cov(y_true, y_pred, ddof=1)[0, 1]

    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom < 1e-12:
        return 1.0
    return float((2.0 * cov_tp) / denom)

def compute_all_metrics(y_true, y_pred):
    """
    Computes MSE, RMSE, MAE, Bias, and Lin's CCC.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ccc = compute_lins_ccc(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "Lin_CCC": ccc
    }

def compute_percentage_gain(baseline_metric, model_metric):
    """
    Computes percentage improvement:
      Gain (%) = ((Baseline - Model) / Baseline) * 100
    Positive gain indicates improvement (reduction in error).
    Negative gain indicates regression.
    """
    return float(((baseline_metric - model_metric) / baseline_metric) * 100.0)

