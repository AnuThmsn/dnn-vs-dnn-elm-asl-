"""
statistical_significance.py
Phase 5: Rigorous Paired Statistical Significance & Bootstrap Confidence Intervals.
Computes paired difference statistics on identical test samples (N=100,000)
using 1,000 bootstrap resamples to establish 95% CIs and two-sided p-values.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_PATH = os.path.join(EXP_DIR, "all_test_predictions.npz")

def compute_paired_bootstrap(y_true, pred_dnn, pred_comp, n_boot=1000, seed=42):
    """
    Computes paired bootstrap distributions for:
    - Delta RMSE = RMSE_dnn - RMSE_comp
    - Delta MAE = MAE_dnn - MAE_comp
    - % Delta RMSE = 100 * (RMSE_dnn - RMSE_comp) / RMSE_dnn
    """
    rng = np.random.default_rng(seed)
    N = len(y_true)

    err_dnn = pred_dnn - y_true
    err_comp = pred_comp - y_true

    se_dnn = err_dnn ** 2
    se_comp = err_comp ** 2
    ae_dnn = np.abs(err_dnn)
    ae_comp = np.abs(err_comp)

    # Point estimates
    rmse_dnn = np.sqrt(np.mean(se_dnn))
    rmse_comp = np.sqrt(np.mean(se_comp))
    mae_dnn = np.mean(ae_dnn)
    mae_comp = np.mean(ae_comp)

    delta_rmse = rmse_dnn - rmse_comp
    delta_mae = mae_dnn - mae_comp
    pct_rmse = 100.0 * delta_rmse / rmse_dnn

    # Paired t-test on squared error and absolute error
    t_stat_se, p_val_se = stats.ttest_rel(se_dnn, se_comp)
    t_stat_ae, p_val_ae = stats.ttest_rel(ae_dnn, ae_comp)

    # Fast chunked bootstrap for N=100,000
    boot_delta_rmse = np.empty(n_boot, dtype=np.float64)
    boot_delta_mae = np.empty(n_boot, dtype=np.float64)
    boot_pct_rmse = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        b_r_dnn = np.sqrt(np.mean(se_dnn[idx]))
        b_r_comp = np.sqrt(np.mean(se_comp[idx]))
        b_m_dnn = np.mean(ae_dnn[idx])
        b_m_comp = np.mean(ae_comp[idx])

        d_r = b_r_dnn - b_r_comp
        boot_delta_rmse[b] = d_r
        boot_delta_mae[b] = b_m_dnn - b_m_comp
        boot_pct_rmse[b] = 100.0 * d_r / b_r_dnn

    ci_delta_rmse = (float(np.percentile(boot_delta_rmse, 2.5)), float(np.percentile(boot_delta_rmse, 97.5)))
    ci_delta_mae = (float(np.percentile(boot_delta_mae, 2.5)), float(np.percentile(boot_delta_mae, 97.5)))
    ci_pct_rmse = (float(np.percentile(boot_pct_rmse, 2.5)), float(np.percentile(boot_pct_rmse, 97.5)))

    return {
        "rmse_comp": float(rmse_comp),
        "mae_comp": float(mae_comp),
        "delta_rmse": float(delta_rmse),
        "ci95_delta_rmse": ci_delta_rmse,
        "delta_mae": float(delta_mae),
        "ci95_delta_mae": ci_delta_mae,
        "pct_rmse_gain": float(pct_rmse),
        "ci95_pct_rmse": ci_pct_rmse,
        "p_val_paired_se": float(p_val_se),
        "p_val_paired_ae": float(p_val_ae)
    }

def main():
    print("=" * 80)
    print("PHASE 5: PAIRED STATISTICAL BOOTSTRAP ANALYSIS (N=100,000 SAMPLES, B=1,000)")
    print("=" * 80)

    preds = np.load(PRED_PATH)
    y_cbf_true = preds["Y_cbf_test"]
    y_att_true = preds["Y_att_test"]
    cbf_dnn = preds["baseline_a_dnn_g_cbf_test"]
    att_dnn = preds["baseline_a_dnn_g_att_test"]

    models_to_test = [
        ("Standalone ELM", "standalone_elm"),
        ("Weighted Ensemble", "weighted_ensemble"),
        ("Residual ELM", "residual_elm"),
        ("DNN-Feature ELM", "dnn_feature_elm"),
        ("Ensemble Feature ELM (5 seeds)", "ensemble_feature_elm_5_seeds")
    ]

    stat_results = []
    print("\nEvaluating paired bootstrap statistics against Baseline A...")
    for label, prefix in models_to_test:
        cbf_m = preds[f"{prefix}_cbf_test"]
        att_m = preds[f"{prefix}_att_test"]

        res_cbf = compute_paired_bootstrap(y_cbf_true, cbf_dnn, cbf_m)
        res_att = compute_paired_bootstrap(y_att_true, att_dnn, att_m)

        stat_results.append({
            "Model": label,
            "Target": "CBF",
            "RMSE": res_cbf["rmse_comp"],
            "Delta_RMSE": res_cbf["delta_rmse"],
            "95%_CI_Delta_RMSE": f"[{res_cbf['ci95_delta_rmse'][0]:.4f}, {res_cbf['ci95_delta_rmse'][1]:.4f}]",
            "Delta_MAE": res_cbf["delta_mae"],
            "95%_CI_Delta_MAE": f"[{res_cbf['ci95_delta_mae'][0]:.4f}, {res_cbf['ci95_delta_mae'][1]:.4f}]",
            "%_RMSE_Gain": f"{res_cbf['pct_rmse_gain']:.2f}% [{res_cbf['ci95_pct_rmse'][0]:.2f}%, {res_cbf['ci95_pct_rmse'][1]:.2f}%]",
            "p_value": f"{res_cbf['p_val_paired_se']:.2e}" if res_cbf['p_val_paired_se'] > 1e-15 else "< 1e-15",
            "Significant": "YES (p<0.001)" if res_cbf['p_val_paired_se'] < 0.001 else "NO"
        })

        stat_results.append({
            "Model": label,
            "Target": "ATT",
            "RMSE": res_att["rmse_comp"],
            "Delta_RMSE": res_att["delta_rmse"],
            "95%_CI_Delta_RMSE": f"[{res_att['ci95_delta_rmse'][0]:.5f}, {res_att['ci95_delta_rmse'][1]:.5f}]",
            "Delta_MAE": res_att["delta_mae"],
            "95%_CI_Delta_MAE": f"[{res_att['ci95_delta_mae'][0]:.5f}, {res_att['ci95_delta_mae'][1]:.5f}]",
            "%_RMSE_Gain": f"{res_att['pct_rmse_gain']:.2f}% [{res_att['ci95_pct_rmse'][0]:.2f}%, {res_att['ci95_pct_rmse'][1]:.2f}%]",
            "p_value": f"{res_att['p_val_paired_se']:.2e}" if res_att['p_val_paired_se'] > 1e-15 else "< 1e-15",
            "Significant": "YES (p<0.001)" if res_att['p_val_paired_se'] < 0.001 else "NO"
        })

    stat_df = pd.DataFrame(stat_results)
    out_csv = os.path.join(EXP_DIR, "statistical_bootstrap_results.csv")
    stat_df.to_csv(out_csv, index=False)
    print(f"\nSaved bootstrap results: {out_csv}")
    print(stat_df.to_string(index=False))

    with open(os.path.join(EXP_DIR, "statistical_bootstrap_results.json"), "w") as f:
        json.dump(stat_results, f, indent=2)

if __name__ == "__main__":
    main()

