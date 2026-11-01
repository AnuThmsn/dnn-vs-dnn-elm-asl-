"""
statistical_analysis.py
Performs paired bootstrap hypothesis testing and confidence interval estimation:
- Paired bootstrap 95% CI for RMSE difference: Delta_RMSE = RMSE_DNN - RMSE_HDELM
- Paired bootstrap 95% CI for MAE difference: Delta_MAE = MAE_DNN - MAE_HDELM
- Paired Wilcoxon signed-rank test on absolute errors (|e_DNN| vs |e_HDELM|)
Fixed seed: 42, 1,000 resamples.
"""

import numpy as np
from scipy import stats

def paired_bootstrap_comparison(y_true, y_pred_dnn, y_pred_hdelm, n_boot=1000, seed=42):
    """
    Computes paired bootstrap distributions for RMSE and MAE differences:
        Delta_RMSE = RMSE(DNN) - RMSE(HDELM)
        Delta_MAE  = MAE(DNN) - MAE(HDELM)
    Positive delta indicates HDELM has lower error than DNN.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_dnn = np.asarray(y_pred_dnn, dtype=np.float64)
    y_pred_hdelm = np.asarray(y_pred_hdelm, dtype=np.float64)
    N = len(y_true)

    err_sq_dnn = (y_true - y_pred_dnn) ** 2
    err_sq_hdelm = (y_true - y_pred_hdelm) ** 2
    err_abs_dnn = np.abs(y_true - y_pred_dnn)
    err_abs_hdelm = np.abs(y_true - y_pred_hdelm)

    orig_rmse_dnn = np.sqrt(np.mean(err_sq_dnn))
    orig_rmse_hdelm = np.sqrt(np.mean(err_sq_hdelm))
    orig_delta_rmse = orig_rmse_dnn - orig_rmse_hdelm

    orig_mae_dnn = np.mean(err_abs_dnn)
    orig_mae_hdelm = np.mean(err_abs_hdelm)
    orig_delta_mae = orig_mae_dnn - orig_mae_hdelm

    rng = np.random.default_rng(seed)
    boot_delta_rmse = np.empty(n_boot, dtype=np.float64)
    boot_delta_mae = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        b_rmse_dnn = np.sqrt(np.mean(err_sq_dnn[idx]))
        b_rmse_hdelm = np.sqrt(np.mean(err_sq_hdelm[idx]))
        boot_delta_rmse[b] = b_rmse_dnn - b_rmse_hdelm

        b_mae_dnn = np.mean(err_abs_dnn[idx])
        b_mae_hdelm = np.mean(err_abs_hdelm[idx])
        boot_delta_mae[b] = b_mae_dnn - b_mae_hdelm

    ci_rmse = np.percentile(boot_delta_rmse, [2.5, 97.5])
    ci_mae = np.percentile(boot_delta_mae, [2.5, 97.5])

    # Empirical two-tailed p-value: fraction crossing zero
    p_rmse = 2.0 * min(np.mean(boot_delta_rmse <= 0), np.mean(boot_delta_rmse >= 0))
    p_rmse = min(max(p_rmse, 1.0 / n_boot), 1.0)

    # Paired Wilcoxon signed-rank test on sample subset (first 10,000 for efficiency)
    sub_n = min(N, 10000)
    w_stat, p_wilcoxon = stats.wilcoxon(err_abs_dnn[:sub_n], err_abs_hdelm[:sub_n], alternative="two-sided")

    return {
        "orig_delta_rmse": float(orig_delta_rmse),
        "ci_rmse_lower": float(ci_rmse[0]),
        "ci_rmse_upper": float(ci_rmse[1]),
        "p_bootstrap_rmse": float(p_rmse),
        "orig_delta_mae": float(orig_delta_mae),
        "ci_mae_lower": float(ci_mae[0]),
        "ci_mae_upper": float(ci_mae[1]),
        "wilcoxon_stat": float(w_stat),
        "p_wilcoxon": float(p_wilcoxon)
    }

