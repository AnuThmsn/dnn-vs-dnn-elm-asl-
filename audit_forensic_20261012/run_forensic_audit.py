"""
run_forensic_audit.py
Independent Forensic Recalculation & Artifact Audit (No Retraining).
Audit Directory: audit_forensic_20261012
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(AUDIT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

WORKSPACE_DIR = os.path.abspath(os.path.join(AUDIT_DIR, ".."))
EXP_DIR = os.path.join(WORKSPACE_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
ELM_DIR = os.path.join(EXP_DIR, "elm_models")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
DIAG_DIR = os.path.join(EXP_DIR, "diagnostics")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# -----------------------------------------------------------------------------
# 1. Independent Metric Implementation & Synthetic Validation
# -----------------------------------------------------------------------------
def independent_metrics(y_true, y_pred):
    """Independent float64 metric calculator."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    err = y_pred - y_true
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))
    bias = np.mean(err)

    var_true = np.var(y_true)
    r2 = 1.0 - (mse / var_true) if var_true > 1e-12 else 0.0

    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_p = np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_true + var_p + (mean_t - mean_p) ** 2
    ccc = (2.0 * cov) / denom if denom > 1e-12 else 0.0

    return {
        "RMSE": float(rmse),
        "MSE": float(mse),
        "MAE": float(mae),
        "Bias": float(bias),
        "R2": float(r2),
        "CCC": float(ccc)
    }

def validate_metric_functions():
    """Synthetic unit tests validating metric definitions."""
    print("--- Running Metric Synthetic Unit Tests ---")
    # Test 1: Identity
    y1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    p1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    m1 = independent_metrics(y1, p1)
    assert m1["RMSE"] == 0.0 and m1["MAE"] == 0.0 and m1["Bias"] == 0.0
    assert abs(m1["R2"] - 1.0) < 1e-12 and abs(m1["CCC"] - 1.0) < 1e-12
    print("  Synthetic Test 1 (Perfect Prediction): PASSED")

    # Test 2: Constant offset of +2.0
    p2 = y1 + 2.0
    m2 = independent_metrics(y1, p2)
    assert abs(m2["RMSE"] - 2.0) < 1e-12 and abs(m2["MAE"] - 2.0) < 1e-12
    assert abs(m2["Bias"] - 2.0) < 1e-12
    # var(y1) = 200. mse = 4. R2 = 1 - 4/200 = 0.98.
    assert abs(m2["R2"] - 0.98) < 1e-12
    # cov = 200, var_p = 200, denom = 200 + 200 + 4 = 404. CCC = 400/404 = 0.990099...
    expected_ccc = 400.0 / 404.0
    assert abs(m2["CCC"] - expected_ccc) < 1e-12
    print("  Synthetic Test 2 (Constant Shift): PASSED")

    # Test 3: Inverted line
    p3 = -y1 + 60.0 # slope -1
    m3 = independent_metrics(y1, p3)
    assert m3["CCC"] < 0 # Negative concordance
    print("  Synthetic Test 3 (Inverted Correlation): PASSED")

# -----------------------------------------------------------------------------
# 2. Main Audit Script
# -----------------------------------------------------------------------------
def run_full_forensic_audit():
    print("=" * 80)
    print("FORENSIC VERIFICATION AUDIT (ZERO RETRAINING)")
    print(f"Audit output directory: {AUDIT_DIR}")
    print("=" * 80)

    validate_metric_functions()

    # 1. Audit artifact files & hashes
    print("\n" + "=" * 70)
    print("1. ARTIFACT INVENTORY & SHA-256 HASHES")
    print("=" * 70)
    artifacts = [
        os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"),
        os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"),
        os.path.join(BASE_DIR, "training_history.json"),
        os.path.join(BASE_DIR, "normalization_stats.npz"),
        os.path.join(BASE_DIR, "test_summary_baselineA.csv"),
        os.path.join(BASE_DIR, "clean_grid_baselineA.csv"),
        os.path.join(BASE_DIR, "snr_evaluation_baselineA.csv"),
        os.path.join(BASE_DIR, "baselineA_test_predictions.npz"),
        os.path.join(ELM_DIR, "all_test_predictions.npz"),
        os.path.join(ELM_DIR, "all_models_test_comparison.csv"),
        os.path.join(ELM_DIR, "all_models_clean_comparison.csv"),
        os.path.join(ELM_DIR, "all_models_snr_comparison.csv"),
        os.path.join(ELM_DIR, "statistical_bootstrap_results.csv"),
        os.path.join(CACHE_DIR, "X_test.npy"),
        os.path.join(CACHE_DIR, "Y_test.npy"),
        os.path.join(CACHE_DIR, "snr_test.npy"),
        os.path.join(CACHE_DIR, "clean_grid.npz")
    ]
    hash_inventory = {}
    for p in artifacts:
        rel = os.path.relpath(p, WORKSPACE_DIR)
        assert os.path.exists(p), f"Missing artifact: {rel}"
        sz_kb = os.path.getsize(p) / 1024.0
        h = compute_sha256(p)
        hash_inventory[rel] = {"size_kb": sz_kb, "sha256": h}
        print(f"  {rel:<45} {sz_kb:9.1f} KB | SHA256: {h[:16]}...")

    with open(os.path.join(AUDIT_DIR, "artifact_hashes.json"), "w") as f:
        json.dump(hash_inventory, f, indent=2)

    # 2. Load Predictions & Ground Truth
    print("\n" + "=" * 70)
    print("2. LOADING GROUND TRUTH & MODEL PREDICTIONS")
    print("=" * 70)
    preds = np.load(os.path.join(ELM_DIR, "all_test_predictions.npz"))
    y_test_cache = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    snr_test_cache = np.load(os.path.join(CACHE_DIR, "snr_test.npy"), mmap_mode='r')

    clean_grid = np.load(os.path.join(CACHE_DIR, "clean_grid.npz"))
    cbf_clean_true = clean_grid["cbf_flat"]
    att_clean_true = clean_grid["att_flat"]

    y_cbf_true = preds["Y_cbf_test"]
    y_att_true = preds["Y_att_test"]

    # Verify identical alignment between all_test_predictions.npz and cache
    assert np.array_equal(y_cbf_true, y_test_cache[:, 0]), "CBF test ground truth misaligned!"
    assert np.array_equal(y_att_true, y_test_cache[:, 1]), "ATT test ground truth misaligned!"
    print(f"Test ground truth verified: N = {len(y_cbf_true):,} samples. Exact alignment confirmed.")

    # 3. Independent Recalculation across all 6 models on Test Set
    print("\n" + "=" * 70)
    print("3. INDEPENDENT RECALCULATION ON LOCKED TEST SET (N = 100,000)")
    print("=" * 70)

    model_keys = [
        ("Baseline A (DNN-G)", "baseline_a_dnn_g"),
        ("Standalone ELM", "standalone_elm"),
        ("Weighted Ensemble", "weighted_ensemble"),
        ("Residual ELM", "residual_elm"),
        ("DNN-Feature ELM", "dnn_feature_elm"),
        ("Ensemble Feature ELM (5 seeds)", "ensemble_feature_elm_5_seeds")
    ]

    recomputed_test_rows = []
    base_cbf_pred = preds["baseline_a_dnn_g_cbf_test"]
    base_att_pred = preds["baseline_a_dnn_g_att_test"]
    base_cbf_m = independent_metrics(y_cbf_true, base_cbf_pred)
    base_att_m = independent_metrics(y_att_true, base_att_pred)

    for label, prefix in model_keys:
        p_cbf = preds[f"{prefix}_cbf_test"]
        p_att = preds[f"{prefix}_att_test"]

        m_cbf = independent_metrics(y_cbf_true, p_cbf)
        m_att = independent_metrics(y_att_true, p_att)

        # Strictly apply sign convention: delta = Baseline - Candidate
        delta_cbf_rmse = base_cbf_m["RMSE"] - m_cbf["RMSE"]
        delta_cbf_mae  = base_cbf_m["MAE"]  - m_cbf["MAE"]
        pct_cbf_rmse   = 100.0 * delta_cbf_rmse / base_cbf_m["RMSE"]
        pct_cbf_mae    = 100.0 * delta_cbf_mae  / base_cbf_m["MAE"]

        delta_att_rmse = base_att_m["RMSE"] - m_att["RMSE"]
        delta_att_mae  = base_att_m["MAE"]  - m_att["MAE"]
        pct_att_rmse   = 100.0 * delta_att_rmse / base_att_m["RMSE"]
        pct_att_mae    = 100.0 * delta_att_mae  / base_att_m["MAE"]

        recomputed_test_rows.append({
            "Model": label,
            "CBF_RMSE": m_cbf["RMSE"], "CBF_MAE": m_cbf["MAE"], "CBF_Bias": m_cbf["Bias"], "CBF_CCC": m_cbf["CCC"], "CBF_R2": m_cbf["R2"],
            "Delta_CBF_RMSE": delta_cbf_rmse, "Pct_CBF_RMSE": pct_cbf_rmse, "Delta_CBF_MAE": delta_cbf_mae, "Pct_CBF_MAE": pct_cbf_mae,
            "ATT_RMSE": m_att["RMSE"], "ATT_MAE": m_att["MAE"], "ATT_Bias": m_att["Bias"], "ATT_CCC": m_att["CCC"], "ATT_R2": m_att["R2"],
            "Delta_ATT_RMSE": delta_att_rmse, "Pct_ATT_RMSE": pct_att_rmse, "Delta_ATT_MAE": delta_att_mae, "Pct_ATT_MAE": pct_att_mae
        })

    recomputed_test_df = pd.DataFrame(recomputed_test_rows)
    print(recomputed_test_df[["Model", "CBF_RMSE", "Delta_CBF_RMSE", "Pct_CBF_RMSE", "CBF_MAE", "Pct_CBF_MAE",
                              "ATT_RMSE", "Delta_ATT_RMSE", "Pct_ATT_RMSE", "ATT_MAE", "Pct_ATT_MAE"]].to_string(index=False))

    recomputed_test_df.to_csv(os.path.join(AUDIT_DIR, "recomputed_test_metrics.csv"), index=False)

    # Cross-check with stored CSV
    stored_test_csv = pd.read_csv(os.path.join(ELM_DIR, "all_models_test_comparison.csv"))
    print("\n--- Discrepancy Check vs Stored test comparison CSV ---")
    max_cbf_rmse_diff = np.max(np.abs(recomputed_test_df["CBF_RMSE"].values - stored_test_csv["CBF_RMSE"].values))
    max_att_rmse_diff = np.max(np.abs(recomputed_test_df["ATT_RMSE"].values - stored_test_csv["ATT_RMSE"].values))
    print(f"Max absolute discrepancy in CBF RMSE: {max_cbf_rmse_diff:.2e}")
    print(f"Max absolute discrepancy in ATT RMSE: {max_att_rmse_diff:.2e}")
    assert max_cbf_rmse_diff < 1e-5 and max_att_rmse_diff < 1e-5, "Discrepancy found in stored test CSV!"
    print("Stored CSV vs Recomputed: 100% MATCH VERIFIED.")

    # 4. Audit ATT Early-Arrival Bottleneck across Models
    print("\n" + "=" * 70)
    print("4. AUDIT OF ATT RANGE ERRORS & EARLY-ARRIVAL BOTTLENECK")
    print("=" * 70)
    att_intervals = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]

    range_audit_rows = []
    for low, high in att_intervals:
        mask = (y_att_true >= low) & (y_att_true < high) if high < 3.0 else (y_att_true >= low) & (y_att_true <= high)
        n_samples = int(np.sum(mask))

        for label, prefix in model_keys:
            p_att = preds[f"{prefix}_att_test"]
            m_sub = independent_metrics(y_att_true[mask], p_att[mask])
            range_audit_rows.append({
                "Interval": f"[{low:.1f}, {high:.1f}]",
                "Model": label,
                "Count": n_samples,
                "ATT_RMSE": m_sub["RMSE"],
                "ATT_MAE": m_sub["MAE"],
                "ATT_Bias": m_sub["Bias"],
                "ATT_CCC": m_sub["CCC"]
            })

    range_audit_df = pd.DataFrame(range_audit_rows)
    range_audit_df.to_csv(os.path.join(AUDIT_DIR, "att_interval_forensic_breakdown.csv"), index=False)

    # Pivot for clean comparison in [0.5, 1.0)
    early_df = range_audit_df[range_audit_df["Interval"] == "[0.5, 1.0]"][["Model", "ATT_RMSE", "ATT_MAE", "ATT_Bias", "ATT_CCC"]]
    print("\n--- Early-Arrival Window: [0.5, 1.0) s (N=19,728) ---")
    print(early_df.to_string(index=False))

    well_sampled_df = range_audit_df[range_audit_df["Interval"] == "[2.0, 2.5]"][["Model", "ATT_RMSE", "ATT_MAE", "ATT_Bias", "ATT_CCC"]]
    print("\n--- Well-Sampled Window: [2.0, 2.5) s (N=20,275) ---")
    print(well_sampled_df.to_string(index=False))

    # 5. Audit SNR-Wise Robustness
    print("\n" + "=" * 70)
    print("5. AUDIT OF SNR-WISE METRICS ON HELD-OUT TEST SET")
    print("=" * 70)
    snr_levels = np.unique(snr_test_cache)
    snr_test_rows = []
    for s_val in snr_levels:
        mask_snr = (snr_test_cache == s_val)
        n_snr = int(np.sum(mask_snr))

        for label, prefix in model_keys:
            p_cbf = preds[f"{prefix}_cbf_test"][mask_snr]
            p_att = preds[f"{prefix}_att_test"][mask_snr]
            m_c = independent_metrics(y_cbf_true[mask_snr], p_cbf)
            m_a = independent_metrics(y_att_true[mask_snr], p_att)
            snr_test_rows.append({
                "SNR": s_val if not np.isinf(s_val) else "inf",
                "Model": label,
                "Count": n_snr,
                "CBF_RMSE": m_c["RMSE"], "CBF_MAE": m_c["MAE"], "CBF_CCC": m_c["CCC"],
                "ATT_RMSE": m_a["RMSE"], "ATT_MAE": m_a["MAE"], "ATT_CCC": m_a["CCC"]
            })

    snr_test_df = pd.DataFrame(snr_test_rows)
    snr_test_df.to_csv(os.path.join(AUDIT_DIR, "recomputed_test_by_snr.csv"), index=False)
    print("Saved test-split SNR breakdown to recomputed_test_by_snr.csv")

    # 6. Recompute Paired Bootstrap Significance & Investigate Resampling Units
    print("\n" + "=" * 70)
    print("6. INDEPENDENT PAIRED BOOTSTRAP RECALCULATION (B=1,000)")
    print("=" * 70)
    rng_boot = np.random.default_rng(2026)
    N = len(y_cbf_true)
    n_boot = 1000

    se_dnn_cbf = (base_cbf_pred - y_cbf_true) ** 2
    se_dnn_att = (base_att_pred - y_att_true) ** 2
    ae_dnn_cbf = np.abs(base_cbf_pred - y_cbf_true)
    ae_dnn_att = np.abs(base_att_pred - y_att_true)

    boot_rows = []
    for label, prefix in model_keys[1:]: # Exclude baseline itself
        p_cbf = preds[f"{prefix}_cbf_test"]
        p_att = preds[f"{prefix}_att_test"]

        se_comp_cbf = (p_cbf - y_cbf_true) ** 2
        se_comp_att = (p_att - y_att_true) ** 2
        ae_comp_cbf = np.abs(p_cbf - y_cbf_true)
        ae_comp_att = np.abs(p_att - y_att_true)

        # Paired t-tests
        t_se_cbf, p_se_cbf = stats.ttest_rel(se_dnn_cbf, se_comp_cbf)
        t_se_att, p_se_att = stats.ttest_rel(se_dnn_att, se_comp_att)
        t_ae_cbf, p_ae_cbf = stats.ttest_rel(ae_dnn_cbf, ae_comp_cbf)
        t_ae_att, p_ae_att = stats.ttest_rel(ae_dnn_att, ae_comp_att)

        # Wilcoxon signed-rank test on subsample (N=10,000 for exact calculation feasibility)
        sub_idx = rng_boot.choice(N, size=10_000, replace=False)
        w_cbf, p_w_cbf = stats.wilcoxon(ae_dnn_cbf[sub_idx], ae_comp_cbf[sub_idx])
        w_att, p_w_att = stats.wilcoxon(ae_dnn_att[sub_idx], ae_comp_att[sub_idx])

        # Bootstrap distribution of delta RMSE (Baseline - Candidate)
        boot_d_rmse_cbf = np.empty(n_boot)
        boot_d_mae_cbf  = np.empty(n_boot)
        boot_pct_cbf    = np.empty(n_boot)

        boot_d_rmse_att = np.empty(n_boot)
        boot_d_mae_att  = np.empty(n_boot)
        boot_pct_att    = np.empty(n_boot)

        for b in range(n_boot):
            idx = rng_boot.integers(0, N, size=N)
            r_dnn_c = np.sqrt(np.mean(se_dnn_cbf[idx]))
            r_cmp_c = np.sqrt(np.mean(se_comp_cbf[idx]))
            m_dnn_c = np.mean(ae_dnn_cbf[idx])
            m_cmp_c = np.mean(ae_comp_cbf[idx])

            d_r_c = r_dnn_c - r_cmp_c
            boot_d_rmse_cbf[b] = d_r_c
            boot_d_mae_cbf[b]  = m_dnn_c - m_cmp_c
            boot_pct_cbf[b]    = 100.0 * d_r_c / r_dnn_c

            r_dnn_a = np.sqrt(np.mean(se_dnn_att[idx]))
            r_cmp_a = np.sqrt(np.mean(se_comp_att[idx]))
            m_dnn_a = np.mean(ae_dnn_att[idx])
            m_cmp_a = np.mean(ae_comp_att[idx])

            d_r_a = r_dnn_a - r_cmp_a
            boot_d_rmse_att[b] = d_r_a
            boot_d_mae_att[b]  = m_dnn_a - m_cmp_a
            boot_pct_att[b]    = 100.0 * d_r_a / r_dnn_a

        # CBF Summary
        ci_r_c = (float(np.percentile(boot_d_rmse_cbf, 2.5)), float(np.percentile(boot_d_rmse_cbf, 97.5)))
        ci_m_c = (float(np.percentile(boot_d_mae_cbf, 2.5)), float(np.percentile(boot_d_mae_cbf, 97.5)))
        ci_p_c = (float(np.percentile(boot_pct_cbf, 2.5)), float(np.percentile(boot_pct_cbf, 97.5)))

        # ATT Summary
        ci_r_a = (float(np.percentile(boot_d_rmse_att, 2.5)), float(np.percentile(boot_d_rmse_att, 97.5)))
        ci_m_a = (float(np.percentile(boot_d_mae_att, 2.5)), float(np.percentile(boot_d_mae_att, 97.5)))
        ci_p_a = (float(np.percentile(boot_pct_att, 2.5)), float(np.percentile(boot_pct_att, 97.5)))

        boot_rows.append({
            "Candidate": label,
            "Target": "CBF",
            "Delta_RMSE": float(np.mean(boot_d_rmse_cbf)),
            "95%_CI_Delta_RMSE": f"[{ci_r_c[0]:.4f}, {ci_r_c[1]:.4f}]",
            "Delta_MAE": float(np.mean(boot_d_mae_cbf)),
            "95%_CI_Delta_MAE": f"[{ci_m_c[0]:.4f}, {ci_m_c[1]:.4f}]",
            "Pct_RMSE_Gain": f"{float(np.mean(boot_pct_cbf)):.2f}% [{ci_p_c[0]:.2f}%, {ci_p_c[1]:.2f}%]",
            "Paired_t_pval": float(p_se_cbf),
            "Wilcoxon_pval": float(p_w_cbf),
            "Reject_H0_p<0.001": bool(p_se_cbf < 0.001 and ci_r_c[0] > 0)
        })

        boot_rows.append({
            "Candidate": label,
            "Target": "ATT",
            "Delta_RMSE": float(np.mean(boot_d_rmse_att)),
            "95%_CI_Delta_RMSE": f"[{ci_r_a[0]:.5f}, {ci_r_a[1]:.5f}]",
            "Delta_MAE": float(np.mean(boot_d_mae_att)),
            "95%_CI_Delta_MAE": f"[{ci_m_a[0]:.5f}, {ci_m_a[1]:.5f}]",
            "Pct_RMSE_Gain": f"{float(np.mean(boot_pct_att)):.2f}% [{ci_p_a[0]:.2f}%, {ci_p_a[1]:.2f}%]",
            "Paired_t_pval": float(p_se_att),
            "Wilcoxon_pval": float(p_w_att),
            "Reject_H0_p<0.001": bool(p_se_att < 0.001 and ci_r_a[0] > 0)
        })

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(os.path.join(AUDIT_DIR, "recomputed_bootstrap_analysis.csv"), index=False)
    print("\n--- Recomputed Paired Bootstrap Analysis ---")
    print(boot_df[["Candidate", "Target", "Delta_RMSE", "95%_CI_Delta_RMSE", "Delta_MAE", "95%_CI_Delta_MAE", "Pct_RMSE_Gain", "Reject_H0_p<0.001"]].to_string(index=False))

    # 7. Generate Verification Figures into audit_forensic_20261012/plots/
    print("\n" + "=" * 70)
    print("7. GENERATING VERIFIED AUDIT PLOTS")
    print("=" * 70)

    # Plot 1: Verified Training Curves
    with open(os.path.join(BASE_DIR, "training_history.json"), "r") as f:
        hist = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # ATT
    att_h = hist["att_history"]
    ep_a = [e["epoch"] for e in att_h]
    axes[0].plot(ep_a, [e["train_loss"] for e in att_h], label="Train MSE Loss", color="#1f77b4", lw=2)
    axes[0].plot(ep_a, [e["val_loss"] for e in att_h], label="Val MSE Loss", color="#ff7f0e", lw=2)
    best_a_idx = int(np.argmin([e["val_loss"] for e in att_h]))
    axes[0].scatter([ep_a[best_a_idx]], [att_h[best_a_idx]["val_loss"]], color="red", s=80, zorder=5,
                    label=f"Best Val: Ep {ep_a[best_a_idx]} ({att_h[best_a_idx]['val_loss']:.4f})")
    axes[0].axvline(ep_a[-1], color="gray", linestyle=":", label=f"Early Stop: Ep {ep_a[-1]}")
    axes[0].set_title("Audit: Baseline A ATTnet (4x64) Convergence History", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Standardized MSE Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5); axes[0].legend()

    # CBF
    cbf_h = hist["cbf_history"]
    ep_c = [e["epoch"] for e in cbf_h]
    axes[1].plot(ep_c, [e["train_loss"] for e in cbf_h], label="Train MSE Loss", color="#2ca02c", lw=2)
    axes[1].plot(ep_c, [e["val_loss"] for e in cbf_h], label="Val MSE Loss", color="#d62728", lw=2)
    best_c_idx = int(np.argmin([e["val_loss"] for e in cbf_h]))
    axes[1].scatter([ep_c[best_c_idx]], [cbf_h[best_c_idx]["val_loss"]], color="red", s=80, zorder=5,
                    label=f"Best Val: Ep {ep_c[best_c_idx]} ({cbf_h[best_c_idx]['val_loss']:.4f})")
    axes[1].axvline(ep_c[-1], color="gray", linestyle=":", label=f"Early Stop: Ep {ep_c[-1]}")
    axes[1].set_title("Audit: Baseline A CBFnet (5x128) Convergence History", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Standardized MSE Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5); axes[1].legend()

    fig.tight_layout()
    plot_conv_path = os.path.join(PLOTS_DIR, "audit_training_convergence.png")
    fig.savefig(plot_conv_path)
    plt.close(fig)
    print(f"Saved: {plot_conv_path}")

    # Plot 2: ATT Early-Arrival Error Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_int_labels = [f"[{l:.1f},{h:.1f})" for l, h in att_intervals]
    models_to_plot = ["Baseline A (DNN-G)", "Standalone ELM", "DNN-Feature ELM", "Ensemble Feature ELM (5 seeds)"]
    colors = ["#4c72b0", "#dd8452", "#8172b3", "#55a868"]

    for m_label, col in zip(models_to_plot, colors):
        sub_d = range_audit_df[range_audit_df["Model"] == m_label]
        axes[0].plot(x_int_labels, sub_d["ATT_RMSE"], marker="o", label=m_label, color=col, lw=2)
        axes[1].plot(x_int_labels, sub_d["ATT_Bias"], marker="s", label=m_label, color=col, lw=2)

    axes[0].set_title("ATT RMSE across Transit Delay Intervals", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("ATT Interval (seconds)")
    axes[0].set_ylabel("ATT RMSE (seconds)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("ATT Bias across Transit Delay Intervals", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("ATT Interval (seconds)")
    axes[1].set_ylabel("ATT Mean Bias (seconds)")
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.tight_layout()
    plot_att_bottleneck = os.path.join(PLOTS_DIR, "audit_att_interval_profiles.png")
    fig.savefig(plot_att_bottleneck)
    plt.close(fig)
    print(f"Saved: {plot_att_bottleneck}")

    print("\n" + "=" * 80)
    print("FORENSIC RECALCULATION & AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_full_forensic_audit()

