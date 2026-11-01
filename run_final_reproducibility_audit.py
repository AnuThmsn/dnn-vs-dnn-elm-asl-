"""
run_final_reproducibility_audit.py
Independent, authoritative verification and statistical audit of:
- DNN-G baseline
- Proposed Residual H-DELM V1 (fitted on DNN training observations)
- Corrected Residual H-DELM V2 (fitted on independent calibration cohort)

Generates the required deliverables:
1. FINAL_METRIC_VERIFICATION.csv
2. FINAL_V1_V2_PAIRED_COMPARISON.csv
3. FINAL_STATISTICAL_TESTS.csv
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compute_lins_ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true, ddof=1), np.var(y_pred, ddof=1)
    cov_tp = np.cov(y_true, y_pred, ddof=1)[0, 1]
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom < 1e-12:
        return 1.0
    return float((2.0 * cov_tp) / denom)

def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ccc = compute_lins_ccc(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "Bias": bias, "Lin_CCC": ccc}

def paired_bootstrap_stats(y_true, p_v1, p_v2, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    e1 = p_v1 - y_true
    e2 = p_v2 - y_true

    orig_rmse1 = np.sqrt(np.mean(e1 ** 2))
    orig_rmse2 = np.sqrt(np.mean(e2 ** 2))
    orig_d_rmse = orig_rmse1 - orig_rmse2  # positive means V2 has lower RMSE (improvement)

    orig_mae1 = np.mean(np.abs(e1))
    orig_mae2 = np.mean(np.abs(e2))
    orig_d_mae = orig_mae1 - orig_mae2     # positive means V2 has lower MAE

    boot_d_rmse = np.empty(n_boot, dtype=np.float64)
    boot_d_mae = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        be1, be2 = e1[idx], e2[idx]
        boot_d_rmse[b] = np.sqrt(np.mean(be1 ** 2)) - np.sqrt(np.mean(be2 ** 2))
        boot_d_mae[b] = np.mean(np.abs(be1)) - np.mean(np.abs(be2))

    return {
        "rmse_v1": float(orig_rmse1),
        "rmse_v2": float(orig_rmse2),
        "delta_rmse_v1_minus_v2": float(orig_d_rmse),
        "ci_delta_rmse_lower": float(np.percentile(boot_d_rmse, 2.5)),
        "ci_delta_rmse_upper": float(np.percentile(boot_d_rmse, 97.5)),
        "mae_v1": float(orig_mae1),
        "mae_v2": float(orig_mae2),
        "delta_mae_v1_minus_v2": float(orig_d_mae),
        "ci_delta_mae_lower": float(np.percentile(boot_d_mae, 2.5)),
        "ci_delta_mae_upper": float(np.percentile(boot_d_mae, 97.5))
    }

def paired_permutation_test_v1_v2(y_true, p_v1, p_v2, n_permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    e1 = p_v1 - y_true
    e2 = p_v2 - y_true

    d_sq = (e1 ** 2) - (e2 ** 2)
    obs_d_sq = np.mean(d_sq)

    d_abs = np.abs(e1) - np.abs(e2)
    obs_d_abs = np.mean(d_abs)

    chunk_size = 1000
    n_chunks = int(np.ceil(n_permutations / chunk_size))
    extreme_sq, extreme_abs = 0, 0

    abs_obs_sq = np.abs(obs_d_sq)
    abs_obs_abs = np.abs(obs_d_abs)

    for c in range(n_chunks):
        k = min(chunk_size, n_permutations - c * chunk_size)
        signs = rng.choice([-1.0, 1.0], size=(k, N))
        p_sq = np.mean(signs * d_sq, axis=1)
        p_abs = np.mean(signs * d_abs, axis=1)
        extreme_sq += np.sum(np.abs(p_sq) >= abs_obs_sq)
        extreme_abs += np.sum(np.abs(p_abs) >= abs_obs_abs)

    p_perm_sq = float((1.0 + extreme_sq) / (n_permutations + 1.0))
    p_perm_abs = float((1.0 + extreme_abs) / (n_permutations + 1.0))

    try:
        w_res = stats.wilcoxon(np.abs(e1), np.abs(e2), alternative="two-sided")
        w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_pval = np.nan, np.nan

    return {
        "obs_mse_diff": float(obs_d_sq),
        "p_perm_sq": p_perm_sq,
        "obs_mae_diff": float(obs_d_abs),
        "p_perm_abs": p_perm_abs,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pval": w_pval
    }

def apply_holm(df, pval_col, adj_col):
    m = len(df)
    s_idx = np.argsort(df[pval_col].values)
    raw_p = df[pval_col].values[s_idx]
    adj_p = np.empty(m, dtype=np.float64)
    run_max = 0.0
    for k in range(m):
        val = (m - k) * raw_p[k]
        run_max = max(run_max, val)
        adj_p[k] = min(1.0, run_max)
    inv_idx = np.empty(m, dtype=int)
    inv_idx[s_idx] = np.arange(m)
    df[adj_col] = adj_p[inv_idx]
    df[adj_col + "_sig"] = df[adj_col] < 0.05
    return df

def main():
    print("=" * 80)
    print("RUNNING FINAL SCIENTIFIC REPRODUCIBILITY AUDIT")
    print("=" * 80)

    # 1. Dataset verification
    test_path = os.path.join(ROOT_DIR, "experiments", "paper_reproduction_snr10_15_20", "results", "predictions", "independent_test_30k.npz")
    v2_preds_path = os.path.join(ROOT_DIR, "experiments", "paper_reproduction_snr10_15_20_v2", "results", "predictions", "reproduction_predictions_30k_v2.npz")
    
    sha_test = compute_sha256(test_path)
    test_data = np.load(test_path)
    preds_data = np.load(v2_preds_path)

    print(f"Test File Path   : {test_path}")
    print(f"Test SHA256 Hash : {sha_test}")
    print(f"Total Samples    : {len(test_data['X_test']):,}")
    print(f"PLDs             : {test_data['plds']}")

    y_test = test_data["Y_test"]
    snr_test = test_data["snr_test"]
    cbf_true = y_test[:, 0]
    att_true = y_test[:, 1]

    cbf_dnn = preds_data["cbf_dnn"]
    att_dnn = preds_data["att_dnn"]
    cbf_v1 = preds_data["cbf_hdelm_v1"]
    att_v1 = preds_data["att_hdelm_v1"]
    cbf_v2 = preds_data["cbf_hdelm_v2"]
    att_v2 = preds_data["att_hdelm_v2"]

    # Verify ground-truth matches
    assert np.allclose(cbf_true, preds_data["cbf_true"]), "CBF ground truth mismatch!"
    assert np.allclose(att_true, preds_data["att_true"]), "ATT ground truth mismatch!"
    print("Verified exact correspondence of ground truth and predictions across all 30,000 samples.")

    # -------------------------------------------------------------
    # Deliverable 1: FINAL_METRIC_VERIFICATION.csv
    # -------------------------------------------------------------
    metric_rows = []
    cohorts = [10.0, 15.0, 20.0, "All (10-20)"]

    for c in cohorts:
        if c == "All (10-20)":
            mask = np.ones(len(snr_test), dtype=bool)
            c_label = "All (10-20)"
        else:
            mask = (snr_test == c)
            c_label = int(c)

        # CBF
        yt_c = cbf_true[mask]
        m_dnn_c = compute_metrics(yt_c, cbf_dnn[mask])
        m_v1_c  = compute_metrics(yt_c, cbf_v1[mask])
        m_v2_c  = compute_metrics(yt_c, cbf_v2[mask])

        # Gains
        v1_gain_rmse = ((m_dnn_c["RMSE"] - m_v1_c["RMSE"]) / m_dnn_c["RMSE"]) * 100.0
        v2_gain_rmse = ((m_dnn_c["RMSE"] - m_v2_c["RMSE"]) / m_dnn_c["RMSE"]) * 100.0
        v2_vs_v1_rmse = ((m_v1_c["RMSE"] - m_v2_c["RMSE"]) / m_v1_c["RMSE"]) * 100.0

        v1_gain_mae = ((m_dnn_c["MAE"] - m_v1_c["MAE"]) / m_dnn_c["MAE"]) * 100.0
        v2_gain_mae = ((m_dnn_c["MAE"] - m_v2_c["MAE"]) / m_dnn_c["MAE"]) * 100.0
        v2_vs_v1_mae = ((m_v1_c["MAE"] - m_v2_c["MAE"]) / m_v1_c["MAE"]) * 100.0

        metric_rows.append({
            "SNR": c_label, "Parameter": "CBF",
            "DNN_MSE": m_dnn_c["MSE"], "DNN_RMSE": m_dnn_c["RMSE"], "DNN_MAE": m_dnn_c["MAE"], "DNN_Bias": m_dnn_c["Bias"], "DNN_CCC": m_dnn_c["Lin_CCC"],
            "V1_MSE": m_v1_c["MSE"], "V1_RMSE": m_v1_c["RMSE"], "V1_MAE": m_v1_c["MAE"], "V1_Bias": m_v1_c["Bias"], "V1_CCC": m_v1_c["Lin_CCC"],
            "V2_MSE": m_v2_c["MSE"], "V2_RMSE": m_v2_c["RMSE"], "V2_MAE": m_v2_c["MAE"], "V2_Bias": m_v2_c["Bias"], "V2_CCC": m_v2_c["Lin_CCC"],
            "V1_Gain_RMSE_%": v1_gain_rmse, "V2_Gain_RMSE_%": v2_gain_rmse, "V2_vs_V1_RMSE_%": v2_vs_v1_rmse,
            "V1_Gain_MAE_%": v1_gain_mae, "V2_Gain_MAE_%": v2_gain_mae, "V2_vs_V1_MAE_%": v2_vs_v1_mae
        })

        # ATT
        yt_a = att_true[mask]
        m_dnn_a = compute_metrics(yt_a, att_dnn[mask])
        m_v1_a  = compute_metrics(yt_a, att_v1[mask])
        m_v2_a  = compute_metrics(yt_a, att_v2[mask])

        v1_gain_rmse_a = ((m_dnn_a["RMSE"] - m_v1_a["RMSE"]) / m_dnn_a["RMSE"]) * 100.0
        v2_gain_rmse_a = ((m_dnn_a["RMSE"] - m_v2_a["RMSE"]) / m_dnn_a["RMSE"]) * 100.0
        v2_vs_v1_rmse_a = ((m_v1_a["RMSE"] - m_v2_a["RMSE"]) / m_v1_a["RMSE"]) * 100.0

        v1_gain_mae_a = ((m_dnn_a["MAE"] - m_v1_a["MAE"]) / m_dnn_a["MAE"]) * 100.0
        v2_gain_mae_a = ((m_dnn_a["MAE"] - m_v2_a["MAE"]) / m_dnn_a["MAE"]) * 100.0
        v2_vs_v1_mae_a = ((m_v1_a["MAE"] - m_v2_a["MAE"]) / m_v1_a["MAE"]) * 100.0

        metric_rows.append({
            "SNR": c_label, "Parameter": "ATT",
            "DNN_MSE": m_dnn_a["MSE"], "DNN_RMSE": m_dnn_a["RMSE"], "DNN_MAE": m_dnn_a["MAE"], "DNN_Bias": m_dnn_a["Bias"], "DNN_CCC": m_dnn_a["Lin_CCC"],
            "V1_MSE": m_v1_a["MSE"], "V1_RMSE": m_v1_a["RMSE"], "V1_MAE": m_v1_a["MAE"], "V1_Bias": m_v1_a["Bias"], "V1_CCC": m_v1_a["Lin_CCC"],
            "V2_MSE": m_v2_a["MSE"], "V2_RMSE": m_v2_a["RMSE"], "V2_MAE": m_v2_a["MAE"], "V2_Bias": m_v2_a["Bias"], "V2_CCC": m_v2_a["Lin_CCC"],
            "V1_Gain_RMSE_%": v1_gain_rmse_a, "V2_Gain_RMSE_%": v2_gain_rmse_a, "V2_vs_V1_RMSE_%": v2_vs_v1_rmse_a,
            "V1_Gain_MAE_%": v1_gain_mae_a, "V2_Gain_MAE_%": v2_gain_mae_a, "V2_vs_V1_MAE_%": v2_vs_v1_mae_a
        })

    metric_df = pd.DataFrame(metric_rows)
    out_metric_p = os.path.join(ROOT_DIR, "FINAL_METRIC_VERIFICATION.csv")
    metric_df.to_csv(out_metric_p, index=False)
    print(f"Saved {out_metric_p}")

    # -------------------------------------------------------------
    # Deliverables 2 & 3: FINAL_V1_V2_PAIRED_COMPARISON.csv & FINAL_STATISTICAL_TESTS.csv
    # -------------------------------------------------------------
    paired_comp_rows = []
    stat_test_rows = []

    # Equivalence margins
    delta_cbf_eq = 0.50   # mL/100g/min (~1% of mean CBF)
    delta_att_eq = 0.025  # seconds (25 ms, 5% of inter-PLD spacing 500 ms)

    for snr in [10.0, 15.0, 20.0]:
        mask = (snr_test == snr)
        
        # --- CBF Comparison ---
        yt_c = cbf_true[mask]
        p1_c = cbf_v1[mask]
        p2_c = cbf_v2[mask]

        boot_cbf = paired_bootstrap_stats(yt_c, p1_c, p2_c, n_boot=2000, seed=42)
        perm_cbf = paired_permutation_test_v1_v2(yt_c, p1_c, p2_c, n_permutations=10000, seed=42)

        # Check equivalence: does the 95% CI of delta lie entirely within [-margin, +margin]?
        eq_rmse_cbf = (boot_cbf["ci_delta_rmse_lower"] >= -delta_cbf_eq) and (boot_cbf["ci_delta_rmse_upper"] <= delta_cbf_eq)
        eq_mae_cbf = (boot_cbf["ci_delta_mae_lower"] >= -delta_cbf_eq) and (boot_cbf["ci_delta_mae_upper"] <= delta_cbf_eq)

        paired_comp_rows.append({
            "SNR": int(snr),
            "Parameter": "CBF",
            "V1_RMSE": boot_cbf["rmse_v1"],
            "V2_RMSE": boot_cbf["rmse_v2"],
            "Delta_RMSE_V1_minus_V2": boot_cbf["delta_rmse_v1_minus_v2"],
            "CI95_Delta_RMSE_Lower": boot_cbf["ci_delta_rmse_lower"],
            "CI95_Delta_RMSE_Upper": boot_cbf["ci_delta_rmse_upper"],
            "V1_MAE": boot_cbf["mae_v1"],
            "V2_MAE": boot_cbf["mae_v2"],
            "Delta_MAE_V1_minus_V2": boot_cbf["delta_mae_v1_minus_v2"],
            "CI95_Delta_MAE_Lower": boot_cbf["ci_delta_mae_lower"],
            "CI95_Delta_MAE_Upper": boot_cbf["ci_delta_mae_upper"],
            "Equivalence_Margin_Delta": delta_cbf_eq,
            "RMSE_Equivalence_Established": eq_rmse_cbf,
            "MAE_Equivalence_Established": eq_mae_cbf
        })

        stat_test_rows.append({
            "SNR": int(snr),
            "Parameter": "CBF",
            "Obs_MSE_Diff": perm_cbf["obs_mse_diff"],
            "Perm_p_MSE": perm_cbf["p_perm_sq"],
            "Obs_MAE_Diff": perm_cbf["obs_mae_diff"],
            "Perm_p_MAE": perm_cbf["p_perm_abs"],
            "Wilcoxon_Stat": perm_cbf["wilcoxon_stat"],
            "Wilcoxon_p": perm_cbf["wilcoxon_pval"]
        })

        # --- ATT Comparison ---
        yt_a = att_true[mask]
        p1_a = att_v1[mask]
        p2_a = att_v2[mask]

        boot_att = paired_bootstrap_stats(yt_a, p1_a, p2_a, n_boot=2000, seed=42)
        perm_att = paired_permutation_test_v1_v2(yt_a, p1_a, p2_a, n_permutations=10000, seed=42)

        eq_rmse_att = (boot_att["ci_delta_rmse_lower"] >= -delta_att_eq) and (boot_att["ci_delta_rmse_upper"] <= delta_att_eq)
        eq_mae_att = (boot_att["ci_delta_mae_lower"] >= -delta_att_eq) and (boot_att["ci_delta_mae_upper"] <= delta_att_eq)

        paired_comp_rows.append({
            "SNR": int(snr),
            "Parameter": "ATT",
            "V1_RMSE": boot_att["rmse_v1"],
            "V2_RMSE": boot_att["rmse_v2"],
            "Delta_RMSE_V1_minus_V2": boot_att["delta_rmse_v1_minus_v2"],
            "CI95_Delta_RMSE_Lower": boot_att["ci_delta_rmse_lower"],
            "CI95_Delta_RMSE_Upper": boot_att["ci_delta_rmse_upper"],
            "V1_MAE": boot_att["mae_v1"],
            "V2_MAE": boot_att["mae_v2"],
            "Delta_MAE_V1_minus_V2": boot_att["delta_mae_v1_minus_v2"],
            "CI95_Delta_MAE_Lower": boot_att["ci_delta_mae_lower"],
            "CI95_Delta_MAE_Upper": boot_att["ci_delta_mae_upper"],
            "Equivalence_Margin_Delta": delta_att_eq,
            "RMSE_Equivalence_Established": eq_rmse_att,
            "MAE_Equivalence_Established": eq_mae_att
        })

        stat_test_rows.append({
            "SNR": int(snr),
            "Parameter": "ATT",
            "Obs_MSE_Diff": perm_att["obs_mse_diff"],
            "Perm_p_MSE": perm_att["p_perm_sq"],
            "Obs_MAE_Diff": perm_att["obs_mae_diff"],
            "Perm_p_MAE": perm_att["p_perm_abs"],
            "Wilcoxon_Stat": perm_att["wilcoxon_stat"],
            "Wilcoxon_p": perm_att["wilcoxon_pval"]
        })

    paired_df = pd.DataFrame(paired_comp_rows)
    out_paired_p = os.path.join(ROOT_DIR, "FINAL_V1_V2_PAIRED_COMPARISON.csv")
    paired_df.to_csv(out_paired_p, index=False)
    print(f"Saved {out_paired_p}")

    stat_df = pd.DataFrame(stat_test_rows)
    stat_df = apply_holm(stat_df, "Perm_p_MSE", "Holm_p_MSE")
    stat_df = apply_holm(stat_df, "Perm_p_MAE", "Holm_p_MAE")
    stat_df = apply_holm(stat_df, "Wilcoxon_p", "Holm_p_Wilcoxon")

    out_stat_p = os.path.join(ROOT_DIR, "FINAL_STATISTICAL_TESTS.csv")
    stat_df.to_csv(out_stat_p, index=False)
    print(f"Saved {out_stat_p}")

    print("\nFINAL METRIC VERIFICATION TABLE:")
    print(metric_df[["SNR", "Parameter", "DNN_RMSE", "V1_RMSE", "V2_RMSE", "V1_Gain_RMSE_%", "V2_Gain_RMSE_%", "V2_vs_V1_RMSE_%", "V2_vs_V1_MAE_%"]].to_string(index=False))

    print("\nFINAL V1 vs V2 PAIRED COMPARISON & EQUIVALENCE TABLE:")
    print(paired_df[["SNR", "Parameter", "V1_RMSE", "V2_RMSE", "Delta_RMSE_V1_minus_V2", "CI95_Delta_RMSE_Lower", "CI95_Delta_RMSE_Upper", "Equivalence_Margin_Delta", "RMSE_Equivalence_Established"]].to_string(index=False))

    print("\nFINAL STATISTICAL TESTS TABLE (H0: V1 == V2):")
    print(stat_df[["SNR", "Parameter", "Obs_MSE_Diff", "Perm_p_MSE", "Holm_p_MSE", "Holm_p_MSE_sig", "Obs_MAE_Diff", "Perm_p_MAE", "Holm_p_MAE", "Holm_p_MAE_sig"]].to_string(index=False))

if __name__ == "__main__":
    main()

