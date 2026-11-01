"""
statistical_testing_v2.py
Comprehensive and rigorous paired statistical hypothesis testing:
1. Paired bootstrap 95% confidence intervals (1,000 resamples) for Delta_RMSE and Delta_MAE.
2. Monte Carlo paired permutation test (sign-flip test, B=10,000) for:
   - Squared-error differences (MSE difference)
   - Absolute-error differences (MAE difference)
3. Paired Wilcoxon signed-rank test on paired absolute errors.
4. Holm-Bonferroni multiplicity correction across family of 6 primary comparisons (3 SNRs x 2 parameters).
"""

import numpy as np
from scipy import stats
import pandas as pd

def paired_bootstrap_ci(y_true, y_pred1, y_pred2, n_boot=1000, seed=42):
    """
    Computes 95% bootstrap confidence intervals for Delta_RMSE and Delta_MAE:
      Delta_RMSE = RMSE(Model 1) - RMSE(Model 2)
      Delta_MAE  = MAE(Model 1)  - MAE(Model 2)
    Positive Delta indicates Model 2 (H-DELM) is superior (lower error).
    """
    rng = np.random.default_rng(seed)
    N = len(y_true)
    
    e1 = y_pred1 - y_true
    e2 = y_pred2 - y_true

    orig_rmse1 = np.sqrt(np.mean(e1 ** 2))
    orig_rmse2 = np.sqrt(np.mean(e2 ** 2))
    orig_delta_rmse = orig_rmse1 - orig_rmse2

    orig_mae1 = np.mean(np.abs(e1))
    orig_mae2 = np.mean(np.abs(e2))
    orig_delta_mae = orig_mae1 - orig_mae2

    boot_delta_rmse = np.empty(n_boot, dtype=np.float64)
    boot_delta_mae = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        b_e1 = e1[idx]
        b_e2 = e2[idx]

        r1 = np.sqrt(np.mean(b_e1 ** 2))
        r2 = np.sqrt(np.mean(b_e2 ** 2))
        boot_delta_rmse[b] = r1 - r2

        m1 = np.mean(np.abs(b_e1))
        m2 = np.mean(np.abs(b_e2))
        boot_delta_mae[b] = m1 - m2

    ci_rmse_lower = float(np.percentile(boot_delta_rmse, 2.5))
    ci_rmse_upper = float(np.percentile(boot_delta_rmse, 97.5))

    ci_mae_lower = float(np.percentile(boot_delta_mae, 2.5))
    ci_mae_upper = float(np.percentile(boot_delta_mae, 97.5))

    return {
        "orig_delta_rmse": float(orig_delta_rmse),
        "ci_rmse_lower": ci_rmse_lower,
        "ci_rmse_upper": ci_rmse_upper,
        "orig_delta_mae": float(orig_delta_mae),
        "ci_mae_lower": ci_mae_lower,
        "ci_mae_upper": ci_mae_upper
    }

def paired_permutation_test(y_true, y_pred1, y_pred2, n_permutations=10000, seed=42):
    """
    Paired sign-flip permutation test for squared-error and absolute-error differences.
    Under H0: error distributions are identical and exchangeable within pairs.
    """
    rng = np.random.default_rng(seed)
    N = len(y_true)

    e1 = y_pred1 - y_true
    e2 = y_pred2 - y_true

    # 1. Squared errors
    d_sq = (e1 ** 2) - (e2 ** 2)
    obs_diff_sq = np.mean(d_sq)

    # 2. Absolute errors
    d_abs = np.abs(e1) - np.abs(e2)
    obs_diff_abs = np.mean(d_abs)

    # Vectorized sign-flip permutation in chunks for efficiency
    chunk_size = 1000
    n_chunks = int(np.ceil(n_permutations / chunk_size))

    count_extreme_sq = 0
    count_extreme_abs = 0

    abs_obs_sq = np.abs(obs_diff_sq)
    abs_obs_abs = np.abs(obs_diff_abs)

    for c in range(n_chunks):
        current_k = min(chunk_size, n_permutations - c * chunk_size)
        # Random signs: shape (current_k, N) with elements in {-1, +1}
        signs = rng.choice([-1.0, 1.0], size=(current_k, N))

        # Permuted mean differences: shape (current_k,)
        perm_mean_sq = np.mean(signs * d_sq, axis=1)
        perm_mean_abs = np.mean(signs * d_abs, axis=1)

        count_extreme_sq += np.sum(np.abs(perm_mean_sq) >= abs_obs_sq)
        count_extreme_abs += np.sum(np.abs(perm_mean_abs) >= abs_obs_abs)

    p_perm_sq = float((1.0 + count_extreme_sq) / (n_permutations + 1.0))
    p_perm_abs = float((1.0 + count_extreme_abs) / (n_permutations + 1.0))

    # 3. Wilcoxon signed-rank test
    try:
        w_res = stats.wilcoxon(np.abs(e1), np.abs(e2), alternative="two-sided")
        w_stat = float(w_res.statistic)
        w_pval = float(w_res.pvalue)
    except Exception as ex:
        w_stat = float("nan")
        w_pval = float("nan")

    return {
        "obs_mse_diff": float(obs_diff_sq),
        "p_perm_sq": p_perm_sq,
        "obs_mae_diff": float(obs_diff_abs),
        "p_perm_abs": p_perm_abs,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval
    }

def apply_holm_bonferroni(df, pval_col="p_perm_abs", adjusted_col="p_perm_abs_holm"):
    """
    Applies the step-down Holm-Bonferroni procedure to control the FWER at alpha = 0.05.
    """
    m = len(df)
    sorted_indices = np.argsort(df[pval_col].values)
    raw_p = df[pval_col].values[sorted_indices]

    adjusted_p = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for k in range(m):
        # Multiplier: (m - k)
        val = (m - k) * raw_p[k]
        running_max = max(running_max, val)
        adjusted_p[k] = min(1.0, running_max)

    # Invert sorting back to original dataframe order
    inv_indices = np.empty(m, dtype=int)
    inv_indices[sorted_indices] = np.arange(m)

    df[adjusted_col] = adjusted_p[inv_indices]
    df[adjusted_col + "_sig"] = df[adjusted_col] < 0.05
    return df

