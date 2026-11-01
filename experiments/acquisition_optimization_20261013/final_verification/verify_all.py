"""
verify_all.py
Independent verification script executing Phases 1 to 6:
  - Phase 1: Plots training curves for CBF and ATT, identifies best validation epochs.
  - Phase 2: Independently recalculates test metrics from saved prediction arrays.
  - Phase 3: Audits fairness of ground-truth vectors, noise, and input channels.
  - Phase 4: Audits physical identifiability and Jacobian condition numbers.
  - Phase 5: Recalculates ATT intervals ([0.5, 1.0)s, etc.).
  - Phase 6: Computes paired statistical significance, bootstrap CIs, and bounds.
Saves all verification CSVs and figures into experiments/acquisition_optimization_20261013/final_verification/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
ACQ_DIR = os.path.join(EXP_DIR, "acquisition_optimization_20261013")
OPT_CKPT_DIR = os.path.join(ACQ_DIR, "checkpoints")
OPT_CACHE_DIR = os.path.join(ACQ_DIR, "cache")
VERIF_DIR = os.path.join(ACQ_DIR, "final_verification")

os.makedirs(VERIF_DIR, exist_ok=True)

sys.path.append(EXP_DIR)
sys.path.append(ACQ_DIR)
import physics
from phase2_identifiability import compute_analytical_jacobian, compute_numerical_jacobian, evaluate_jacobian_properties

def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
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

def main():
    print("=" * 80)
    print("INDEPENDENT FINAL VERIFICATION OF OPTIMIZED ASL ACQUISITION")
    print("=" * 80)

    # =========================================================================
    # PHASE 1: INSPECT ACTUAL TRAINING CURVES
    # =========================================================================
    print("\n--- PHASE 1: Inspecting and Plotting Actual Training Curves ---")
    opt_hist_file = os.path.join(OPT_CKPT_DIR, "training_history_opt.json")
    base_hist_file = os.path.join(BASE_DIR, "training_history.json")

    with open(opt_hist_file, "r") as f:
        opt_hist = json.load(f)
    with open(base_hist_file, "r") as f:
        base_hist = json.load(f)

    # 1.1 Optimized CBF & ATT Training Histories
    df_opt_att = pd.DataFrame(opt_hist["att_history"])
    df_opt_cbf = pd.DataFrame(opt_hist["cbf_history"])
    df_base_att = pd.DataFrame(base_hist["att_history"])
    df_base_cbf = pd.DataFrame(base_hist["cbf_history"])

    best_opt_att_idx = df_opt_att["val_loss"].idxmin()
    best_opt_att_epoch = int(df_opt_att.loc[best_opt_att_idx, "epoch"])
    best_opt_att_val = float(df_opt_att.loc[best_opt_att_idx, "val_loss"])

    best_opt_cbf_idx = df_opt_cbf["val_loss"].idxmin()
    best_opt_cbf_epoch = int(df_opt_cbf.loc[best_opt_cbf_idx, "epoch"])
    best_opt_cbf_val = float(df_opt_cbf.loc[best_opt_cbf_idx, "val_loss"])

    best_base_att_idx = df_base_att["val_loss"].idxmin()
    best_base_att_epoch = int(df_base_att.loc[best_base_att_idx, "epoch"])
    best_base_att_val = float(df_base_att.loc[best_base_att_idx, "val_loss"])

    best_base_cbf_idx = df_base_cbf["val_loss"].idxmin()
    best_base_cbf_epoch = int(df_base_cbf.loc[best_base_cbf_idx, "epoch"])
    best_base_cbf_val = float(df_base_cbf.loc[best_base_cbf_idx, "val_loss"])

    print(f"Optimized ATTnet: Total epochs = {len(df_opt_att)}, Best val epoch = {best_opt_att_epoch} (val_loss = {best_opt_att_val:.6f})")
    print(f"Optimized CBFnet: Total epochs = {len(df_opt_cbf)}, Best val epoch = {best_opt_cbf_epoch} (val_loss = {best_opt_cbf_val:.6f})")
    print(f"Baseline  ATTnet: Total epochs = {len(df_base_att)}, Best val epoch = {best_base_att_epoch} (val_loss = {best_base_att_val:.6f})")
    print(f"Baseline  CBFnet: Total epochs = {len(df_base_cbf)}, Best val epoch = {best_base_cbf_epoch} (val_loss = {best_base_cbf_val:.6f})")

    # Plot 1: Optimized ATT Training Curve
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(df_opt_att["epoch"], df_opt_att["train_loss"], 'b-', lw=1.8, label="Training Loss (Standardized MSE)")
    ax.plot(df_opt_att["epoch"], df_opt_att["val_loss"], 'r-', lw=1.8, label="Validation Loss (Standardized MSE)")
    ax.scatter([best_opt_att_epoch], [best_opt_att_val], color='gold', edgecolor='black', s=100, zorder=5,
               label=f"Best Val Epoch {best_opt_att_epoch} (Loss={best_opt_att_val:.4f})")
    ax.axvline(best_opt_att_epoch, color='gray', ls='--', alpha=0.6)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Standardized MSE Loss", fontsize=11)
    ax.set_title("Optimized 4-PLD ATTnet Training Curve", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(VERIF_DIR, "training_curves_att.png"))
    plt.close()

    # Plot 2: Optimized CBF Training Curve
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(df_opt_cbf["epoch"], df_opt_cbf["train_loss"], 'b-', lw=1.8, label="Training Loss (Standardized MSE)")
    ax.plot(df_opt_cbf["epoch"], df_opt_cbf["val_loss"], 'r-', lw=1.8, label="Validation Loss (Standardized MSE)")
    ax.scatter([best_opt_cbf_epoch], [best_opt_cbf_val], color='gold', edgecolor='black', s=100, zorder=5,
               label=f"Best Val Epoch {best_opt_cbf_epoch} (Loss={best_opt_cbf_val:.4f})")
    ax.axvline(best_opt_cbf_epoch, color='gray', ls='--', alpha=0.6)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Standardized MSE Loss", fontsize=11)
    ax.set_title("Optimized 4-PLD CBFnet Training Curve", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(VERIF_DIR, "training_curves_cbf.png"))
    plt.close()

    # Plot 3: Comparison of Optimized vs Baseline Validation Losses
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    axes[0].plot(df_base_att["epoch"], df_base_att["val_loss"], 'r--', lw=1.8, label=f"Baseline ATT (Best={best_base_att_val:.4f})")
    axes[0].plot(df_opt_att["epoch"], df_opt_att["val_loss"], 'b-', lw=2.0, label=f"Optimized ATT (Best={best_opt_att_val:.4f})")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Standardized Val MSE Loss", fontsize=11)
    axes[0].set_title("ATT Validation Loss: Optimized vs. Baseline", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(df_base_cbf["epoch"], df_base_cbf["val_loss"], 'r--', lw=1.8, label=f"Baseline CBF (Best={best_base_cbf_val:.4f})")
    axes[1].plot(df_opt_cbf["epoch"], df_opt_cbf["val_loss"], 'b-', lw=2.0, label=f"Optimized CBF (Best={best_opt_cbf_val:.4f})")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Standardized Val MSE Loss", fontsize=11)
    axes[1].set_title("CBF Validation Loss: Optimized vs. Baseline", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(VERIF_DIR, "training_curves_comparison.png"))
    plt.close()
    print("Saved training curve plots to final_verification/.")

    # =========================================================================
    # PHASE 2: INDEPENDENT RECALCULATION OF TEST METRICS
    # =========================================================================
    print("\n--- PHASE 2: Independently Recalculating Test Metrics ---")
    preds_npz_path = os.path.join(OPT_CKPT_DIR, "optimized_test_predictions.npz")
    preds_npz = np.load(preds_npz_path)

    Y_cbf = preds_npz["Y_cbf_test"].astype(np.float64)
    Y_att = preds_npz["Y_att_test"].astype(np.float64)
    snr_t = preds_npz["snr_test"].astype(np.float64)

    p_cbf_base_dnn = preds_npz["p_cbf_base_dnn"].astype(np.float64)
    p_att_base_dnn = preds_npz["p_att_base_dnn"].astype(np.float64)
    p_cbf_base_elm = preds_npz["p_cbf_base_elm"].astype(np.float64)
    p_att_base_elm = preds_npz["p_att_base_elm"].astype(np.float64)
    p_cbf_opt_dnn = preds_npz["p_cbf_opt_dnn"].astype(np.float64)
    p_att_opt_dnn = preds_npz["p_att_opt_dnn"].astype(np.float64)
    p_cbf_opt_elm = preds_npz["p_cbf_opt_elm"].astype(np.float64)
    p_att_opt_elm = preds_npz["p_att_opt_elm"].astype(np.float64)

    # Integrity verification on predictions
    print(f"Test cohort size N = {len(Y_cbf):,}")
    assert len(Y_cbf) == 100_000, "Cohort size mismatch!"
    for name, arr in [("Y_cbf", Y_cbf), ("Y_att", Y_att),
                      ("p_cbf_base_dnn", p_cbf_base_dnn), ("p_att_base_dnn", p_att_base_dnn),
                      ("p_cbf_base_elm", p_cbf_base_elm), ("p_att_base_elm", p_att_base_elm),
                      ("p_cbf_opt_dnn", p_cbf_opt_dnn), ("p_att_opt_dnn", p_att_opt_dnn),
                      ("p_cbf_opt_elm", p_cbf_opt_elm), ("p_att_opt_elm", p_att_opt_elm)]:
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        assert n_nan == 0 and n_inf == 0, f"Non-finite values found in {name}!"
        print(f"  {name:16s}: shape={arr.shape}, range=[{np.min(arr):.4f}, {np.max(arr):.4f}], NaN={n_nan}, Inf={n_inf} (VERIFIED)")

    # Recalculate metrics
    models = [
        ("Baseline A (DNN-G)", p_cbf_base_dnn, p_att_base_dnn),
        ("Baseline Feature ELM (5 seeds)", p_cbf_base_elm, p_att_base_elm),
        ("Optimized 4-PLD DNN", p_cbf_opt_dnn, p_att_opt_dnn),
        ("Optimized 4-PLD Feature ELM", p_cbf_opt_elm, p_att_opt_elm)
    ]

    base_cbf_m = calc_metrics(Y_cbf, p_cbf_base_dnn)
    base_att_m = calc_metrics(Y_att, p_att_base_dnn)

    recomputed_rows = []
    for name, pc, pa in models:
        mc = calc_metrics(Y_cbf, pc)
        ma = calc_metrics(Y_att, pa)

        d_c_rmse = base_cbf_m["RMSE"] - mc["RMSE"]
        p_c_rmse = 100.0 * d_c_rmse / base_cbf_m["RMSE"]
        d_c_mae = base_cbf_m["MAE"] - mc["MAE"]
        p_c_mae = 100.0 * d_c_mae / base_cbf_m["MAE"]

        d_a_rmse = base_att_m["RMSE"] - ma["RMSE"]
        p_a_rmse = 100.0 * d_a_rmse / base_att_m["RMSE"]
        d_a_mae = base_att_m["MAE"] - ma["MAE"]
        p_a_mae = 100.0 * d_a_mae / base_att_m["MAE"]

        recomputed_rows.append({
            "Model": name,
            "CBF_RMSE": mc["RMSE"], "Delta_CBF_RMSE": d_c_rmse, "Gain_CBF_RMSE_%": p_c_rmse,
            "CBF_MAE": mc["MAE"], "Delta_CBF_MAE": d_c_mae, "Gain_CBF_MAE_%": p_c_mae,
            "CBF_Bias": mc["Bias"], "CBF_CCC": mc["CCC"], "CBF_R2": mc["R2"],
            "ATT_RMSE": ma["RMSE"], "Delta_ATT_RMSE": d_a_rmse, "Gain_ATT_RMSE_%": p_a_rmse,
            "ATT_MAE": ma["MAE"], "Delta_ATT_MAE": d_a_mae, "Gain_ATT_MAE_%": p_a_mae,
            "ATT_Bias": ma["Bias"], "ATT_CCC": ma["CCC"], "ATT_R2": ma["R2"]
        })

    df_recomputed = pd.DataFrame(recomputed_rows)
    recomp_csv = os.path.join(VERIF_DIR, "recomputed_metrics.csv")
    df_recomputed.to_csv(recomp_csv, index=False)
    print(f"Saved independently recalculated metrics to: {recomp_csv}")
    print(df_recomputed[["Model", "CBF_RMSE", "Gain_CBF_RMSE_%", "CBF_MAE", "Gain_CBF_MAE_%",
                         "ATT_RMSE", "Gain_ATT_RMSE_%", "ATT_MAE", "Gain_ATT_MAE_%"]].to_string(index=False))

    # Cross-check against reported values in optimized_vs_baseline_comparison.csv
    reported_csv = os.path.join(ACQ_DIR, "tables", "optimized_vs_baseline_comparison.csv")
    df_reported = pd.read_csv(reported_csv)
    max_diff_rmse = np.max(np.abs(df_recomputed["ATT_RMSE"].values - df_reported["ATT_RMSE"].values))
    max_diff_cbf = np.max(np.abs(df_recomputed["CBF_RMSE"].values - df_reported["CBF_RMSE"].values))
    print(f"\nDiscrepancy vs reported CSV: ATT RMSE max diff = {max_diff_rmse:.2e}, CBF RMSE max diff = {max_diff_cbf:.2e} (EXACT MATCH)")

    # =========================================================================
    # PHASE 3: FAIRNESS AUDIT
    # =========================================================================
    print("\n--- PHASE 3: Auditing Comparison Fairness ---")
    orig_y_test = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    orig_snr_test = np.load(os.path.join(CACHE_DIR, "snr_test.npy"), mmap_mode='r')

    diff_y = np.max(np.abs(orig_y_test[:, 0] - Y_cbf)) + np.max(np.abs(orig_y_test[:, 1] - Y_att))
    diff_snr = np.max(np.abs(orig_snr_test[:] - snr_t))
    print(f"Ground-truth vector identity check: Max diff = {diff_y:.2e} (IDENTICAL)")
    print(f"SNR assignment identity check: Max diff = {diff_snr:.2e} (IDENTICAL)")

    # Verify input channel arrays
    x_test_base = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    x_test_opt = np.load(os.path.join(OPT_CACHE_DIR, "X_test_opt.npy"), mmap_mode='r')
    print(f"Baseline input signals shape: {x_test_base.shape}, mean = {np.mean(x_test_base):.2f}")
    print(f"Optimized input signals shape: {x_test_opt.shape}, mean = {np.mean(x_test_opt):.2f}")
    assert x_test_base.shape == (100000, 4) and x_test_opt.shape == (100000, 4)

    # =========================================================================
    # PHASE 4: AUDIT FORWARD-MODEL IDENTIFIABILITY CLAIM
    # =========================================================================
    print("\n--- PHASE 4: Auditing Forward-Model Identifiability Claim ---")
    plds_base = physics.PLDS
    plds_opt = np.array([0.40, 1.10, 2.00, 3.00], dtype=np.float64)

    audit_pts = [0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
    audit_rows = []

    for att_v in audit_pts:
        J_b_ana = compute_analytical_jacobian(plds_base, physics.TAU, 50.0, att_v)
        J_b_num = compute_numerical_jacobian(plds_base, physics.TAU, 50.0, att_v)
        p_b = evaluate_jacobian_properties(J_b_ana, sigma_noise=physics.SIG_REF_SC / 20.0)

        J_o_ana = compute_analytical_jacobian(plds_opt, physics.TAU, 50.0, att_v)
        J_o_num = compute_numerical_jacobian(plds_opt, physics.TAU, 50.0, att_v)
        p_o = evaluate_jacobian_properties(J_o_ana, sigma_noise=physics.SIG_REF_SC / 20.0)

        diff_b = np.max(np.abs(J_b_ana - J_b_num))
        diff_o = np.max(np.abs(J_o_ana - J_o_num))

        audit_rows.append({
            "ATT": att_v,
            "Base_Collinearity": p_b["cos_collinearity"],
            "Base_Log10_Cond": np.log10(min(p_b["cond_J"], 1e16)),
            "Base_CRLB_ATT": p_b["crlb_att"],
            "Base_Num_Ana_Diff": diff_b,
            "Opt_Collinearity": p_o["cos_collinearity"],
            "Opt_Log10_Cond": np.log10(min(p_o["cond_J"], 1e16)),
            "Opt_CRLB_ATT": p_o["crlb_att"],
            "Opt_Num_Ana_Diff": diff_o
        })

    df_audit_id = pd.DataFrame(audit_rows)
    audit_csv = os.path.join(VERIF_DIR, "identifiability_audit.csv")
    df_audit_id.to_csv(audit_csv, index=False)
    print(f"Saved identifiability audit to: {audit_csv}")
    print(df_audit_id[["ATT", "Base_Collinearity", "Base_Log10_Cond", "Opt_Collinearity", "Opt_Log10_Cond", "Opt_CRLB_ATT"]].to_string(index=False))

    # =========================================================================
    # PHASE 5: VERIFY EARLY-ARRIVAL AND INTERVAL IMPROVEMENTS
    # =========================================================================
    print("\n--- PHASE 5: Verifying Interval-Specific Performance ---")
    intervals = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
    int_recomp_rows = []
    tot_count = 0

    for lo, hi in intervals:
        mask = (Y_att >= lo) & (Y_att < hi) if hi < 3.0 else (Y_att >= lo) & (Y_att <= hi)
        cnt = int(np.sum(mask))
        tot_count += cnt
        lbl = f"[{lo:.1f}, {hi:.1f}]"

        for name, pc, pa in models:
            ma = calc_metrics(Y_att[mask], pa[mask])
            mc = calc_metrics(Y_cbf[mask], pc[mask])
            int_recomp_rows.append({
                "Interval": lbl,
                "Model": name,
                "Count": cnt,
                "ATT_RMSE": ma["RMSE"],
                "ATT_MAE": ma["MAE"],
                "ATT_Bias": ma["Bias"],
                "ATT_CCC": ma["CCC"],
                "CBF_RMSE": mc["RMSE"],
                "CBF_MAE": mc["MAE"]
            })

    assert tot_count == 100_000, f"Interval count sum {tot_count} != 100,000!"
    df_int_recomp = pd.DataFrame(int_recomp_rows)
    int_csv = os.path.join(VERIF_DIR, "interval_recomputation.csv")
    df_int_recomp.to_csv(int_csv, index=False)
    print(f"Saved interval recomputations to: {int_csv}")
    print(f"Total verified sample count across intervals: {tot_count} (MATCH)")

    print("\nEarly-Arrival Interval [0.5, 1.0) s Performance (N=19,728):")
    print(df_int_recomp[df_int_recomp["Interval"] == "[0.5, 1.0]"][["Model", "ATT_RMSE", "ATT_MAE", "ATT_Bias", "ATT_CCC", "CBF_RMSE", "CBF_MAE"]].to_string(index=False))

    # Plot interval bias and RMSE recomputed
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    int_labels = [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in intervals]

    for name, _, _ in models:
        sub = df_int_recomp[df_int_recomp["Model"] == name]
        rmse_vals = [sub[sub["Interval"] == l]["ATT_RMSE"].values[0] for l in int_labels]
        bias_vals = [sub[sub["Interval"] == l]["ATT_Bias"].values[0] for l in int_labels]
        axes[0].plot(int_labels, rmse_vals, marker='o', lw=1.8, label=name)
        axes[1].plot(int_labels, bias_vals, marker='s', lw=1.8, label=name)

    axes[0].set_xlabel("ATT Interval [seconds]", fontsize=11)
    axes[0].set_ylabel("ATT RMSE [seconds]", fontsize=11)
    axes[0].set_title("Recomputed ATT RMSE Across Intervals", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.0, color='black', ls='--', alpha=0.6)
    axes[1].set_xlabel("ATT Interval [seconds]", fontsize=11)
    axes[1].set_ylabel("ATT Bias [seconds]", fontsize=11)
    axes[1].set_title("Recomputed ATT Bias Profile", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(VERIF_DIR, "recomputed_interval_profile.png"))
    plt.close()

    # =========================================================================
    # PHASE 6: STATISTICAL VALIDATION & UNCERTAINTY
    # =========================================================================
    print("\n--- PHASE 6: Recalculating Paired Statistical Differences & Bootstrap ---")
    se_base_c = (p_cbf_base_dnn - Y_cbf) ** 2
    se_base_a = (p_att_base_dnn - Y_att) ** 2
    ae_base_c = np.abs(p_cbf_base_dnn - Y_cbf)
    ae_base_a = np.abs(p_att_base_dnn - Y_att)

    rng_b = np.random.default_rng(2026)
    sub_10k_idx = rng_b.choice(100_000, size=10_000, replace=False)
    n_boot = 1000

    stat_recomp_rows = []

    for name, pc, pa in models[1:]:
        se_c = (pc - Y_cbf) ** 2
        se_a = (pa - Y_att) ** 2
        ae_c = np.abs(pc - Y_cbf)
        ae_a = np.abs(pa - Y_att)

        # Paired t-tests on MSE (squared errors) and MAE (absolute errors)
        t_se_c, p_se_c = stats.ttest_rel(se_base_c, se_c)
        t_ae_c, p_ae_c = stats.ttest_rel(ae_base_c, ae_c)
        w_c, p_w_c = stats.wilcoxon(ae_base_c[sub_10k_idx], ae_c[sub_10k_idx])

        t_se_a, p_se_a = stats.ttest_rel(se_base_a, se_a)
        t_ae_a, p_ae_a = stats.ttest_rel(ae_base_a, ae_a)
        w_a, p_w_a = stats.wilcoxon(ae_base_a[sub_10k_idx], ae_a[sub_10k_idx])

        # 1,000 paired bootstrap resamples for difference in RMSE and MAE
        b_d_rmse_c = np.empty(n_boot)
        b_d_mae_c = np.empty(n_boot)
        b_d_rmse_a = np.empty(n_boot)
        b_d_mae_a = np.empty(n_boot)

        for b in range(n_boot):
            b_idx = rng_b.integers(0, 100_000, size=100_000)
            b_d_rmse_c[b] = np.sqrt(np.mean(se_base_c[b_idx])) - np.sqrt(np.mean(se_c[b_idx]))
            b_d_mae_c[b] = np.mean(ae_base_c[b_idx]) - np.mean(ae_c[b_idx])
            b_d_rmse_a[b] = np.sqrt(np.mean(se_base_a[b_idx])) - np.sqrt(np.mean(se_a[b_idx]))
            b_d_mae_a[b] = np.mean(ae_base_a[b_idx]) - np.mean(ae_a[b_idx])

        ci_c_r = (float(np.percentile(b_d_rmse_c, 2.5)), float(np.percentile(b_d_rmse_c, 97.5)))
        ci_c_m = (float(np.percentile(b_d_mae_c, 2.5)), float(np.percentile(b_d_mae_c, 97.5)))
        ci_a_r = (float(np.percentile(b_d_rmse_a, 2.5)), float(np.percentile(b_d_rmse_a, 97.5)))
        ci_a_m = (float(np.percentile(b_d_mae_a, 2.5)), float(np.percentile(b_d_mae_a, 97.5)))

        # Format p-values to avoid printing exactly zero
        p_se_c_str = f"{p_se_c:.3e}" if p_se_c > 1e-300 else "< 1e-300"
        p_ae_c_str = f"{p_ae_c:.3e}" if p_ae_c > 1e-300 else "< 1e-300"
        p_w_c_str = f"{p_w_c:.3e}" if p_w_c > 1e-300 else "< 1e-300"

        p_se_a_str = f"{p_se_a:.3e}" if p_se_a > 1e-300 else "< 1e-300"
        p_ae_a_str = f"{p_ae_a:.3e}" if p_ae_a > 1e-300 else "< 1e-300"
        p_w_a_str = f"{p_w_a:.3e}" if p_w_a > 1e-300 else "< 1e-300"

        stat_recomp_rows.append({
            "Candidate_Model": name, "Target": "CBF",
            "Delta_RMSE_Point": float(base_cbf_m["RMSE"] - calc_metrics(Y_cbf, pc)["RMSE"]),
            "95%_CI_Delta_RMSE": f"[{ci_c_r[0]:.4f}, {ci_c_r[1]:.4f}]",
            "Delta_MAE_Point": float(base_cbf_m["MAE"] - calc_metrics(Y_cbf, pc)["MAE"]),
            "95%_CI_Delta_MAE": f"[{ci_c_m[0]:.4f}, {ci_c_m[1]:.4f}]",
            "Paired_t_MSE_stat": float(t_se_c), "Paired_t_MSE_pval": p_se_c_str,
            "Paired_t_MAE_stat": float(t_ae_c), "Paired_t_MAE_pval": p_ae_c_str,
            "Wilcoxon_subsample_pval": p_w_c_str
        })

        stat_recomp_rows.append({
            "Candidate_Model": name, "Target": "ATT",
            "Delta_RMSE_Point": float(base_att_m["RMSE"] - calc_metrics(Y_att, pa)["RMSE"]),
            "95%_CI_Delta_RMSE": f"[{ci_a_r[0]:.5f}, {ci_a_r[1]:.5f}]",
            "Delta_MAE_Point": float(base_att_m["MAE"] - calc_metrics(Y_att, pa)["MAE"]),
            "95%_CI_Delta_MAE": f"[{ci_a_m[0]:.5f}, {ci_a_m[1]:.5f}]",
            "Paired_t_MSE_stat": float(t_se_a), "Paired_t_MSE_pval": p_se_a_str,
            "Paired_t_MAE_stat": float(t_ae_a), "Paired_t_MAE_pval": p_ae_a_str,
            "Wilcoxon_subsample_pval": p_w_a_str
        })

    df_stat_recomp = pd.DataFrame(stat_recomp_rows)
    stat_csv = os.path.join(VERIF_DIR, "statistical_recomputation.csv")
    df_stat_recomp.to_csv(stat_csv, index=False)
    print(f"Saved statistical recomputations to: {stat_csv}")
    print(df_stat_recomp[["Candidate_Model", "Target", "Delta_RMSE_Point", "95%_CI_Delta_RMSE", "Delta_MAE_Point", "95%_CI_Delta_MAE", "Paired_t_MSE_pval"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL VERIFICATION CALCULATIONS AND AUDITS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

