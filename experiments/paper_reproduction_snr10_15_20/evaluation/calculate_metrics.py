"""
calculate_metrics.py
Strict, mathematically verified implementation of evaluation metrics:
- MSE
- RMSE
- MAE
- Lin's Concordance Correlation Coefficient (CCC)
- Percentage Improvement
"""

import numpy as np

def compute_all_metrics(y_true, y_pred):
    """
    Computes MSE, RMSE, MAE, and Lin's Concordance Correlation Coefficient (CCC).
    Uses population-consistent (ddof=0) variance and covariance definitions.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    
    # Population variances and covariance for exact Lin's CCC
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_t = np.var(y_true, ddof=0)
    var_p = np.var(y_pred, ddof=0)
    cov_tp = np.mean((y_true - mean_t) * (y_pred - mean_p))
    
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom > 1e-12:
        ccc = float(2.0 * cov_tp / denom)
    else:
        ccc = 1.0 if np.allclose(y_true, y_pred) else 0.0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "Lin_CCC": ccc
    }

def compute_percentage_gain(metric_dnn, metric_hdelm):
    """
    Percentage improvement in RMSE or MAE of H-DELM relative to DNN-G:
        Gain (%) = ((Metric_DNN - Metric_HDELM) / Metric_DNN) * 100
    Positive means H-DELM improves over DNN-G; negative means regression.
    """
    return ((metric_dnn - metric_hdelm) / metric_dnn) * 100.0

