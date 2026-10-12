"""
run_locked_final_evaluation.py
Phase 6: Locked Final Evaluation on the Held-Out Test Cohort (N=100,000).
Compares:
  1. Baseline A (DNN-G) [Fixed 4-PLD: 1.525, 2.025, 2.525, 3.025 s]
  2. Baseline Ensemble Feature ELM [Fixed 4-PLD]
  3. Optimized 4-PLD DNN [0.40, 1.10, 2.00, 3.00 s]
  4. Optimized 4-PLD Feature ELM (5 seeds) [0.40, 1.10, 2.00, 3.00 s]
Computes:
  - Global metrics (RMSE, MAE, Bias, R2, CCC) and relative percentage gains
  - Paired t-tests, Wilcoxon signed-rank tests, 1,000-replicate bootstrap 95% CIs
  - SNR stratification across 12 SNR tiers
  - ATT interval breakdown across 5 physiological windows (prominently [0.5, 1.0) s)
Saves:
  - tables/optimized_vs_baseline_comparison.csv
  - tables/att_interval_comparison.csv
  - checkpoints/optimized_test_predictions.npz
  - plots/optimized_vs_baseline_error_bars.png
  - plots/att_interval_bias_and_rmse.png
  - plots/snr_stratified_gain_curves.png
"""

import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
ELM_DIR = os.path.join(EXP_DIR, "elm_models")
OUT_DIR = os.path.join(EXP_DIR, "acquisition_optimization_20261013")
OPT_CACHE_DIR = os.path.join(OUT_DIR, "cache")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

for d in [TABLES_DIR, PLOTS_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.append(EXP_DIR)
sys.path.append(ELM_DIR)
sys.path.append(OUT_DIR)
import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

def main():
    print("=" * 80)
    print("PHASE 6: HONEST LOCKED FINAL EVALUATION ON TEST COHORT (N=100,000)")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Ground Truth and SNR
    y_test_mmap = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    snr_test = np.load(os.path.join(CACHE_DIR, "snr_test.npy"), mmap_mode='r')
    Y_cbf_test = y_test_mmap[:, 0].astype(np.float64)
    Y_att_test = y_test_mmap[:, 1].astype(np.float64)

    # 2. Load Baseline Predictions (Candidates 1 & 2)
    print("\nLoading Baseline Reference Predictions (N=100,000)...")
    base_preds_path = os.path.join(EXP_DIR, "significant_improvement_20261012", "checkpoints", "finalists_test_predictions.npz")
    if os.path.exists(base_preds_path):
        base_npz = np.load(base_preds_path)
        p_cbf_base_dnn = base_npz["p_cbf_base_test"]
        p_att_base_dnn = base_npz["p_att_base_test"]
        p_cbf_base_elm = base_npz["p_cbf_ens_test"]
        p_att_base_elm = base_npz["p_att_ens_test"]
    else:
        # Re-predict using saved Baseline models
        norm_base = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
        x_test_base = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
        X_test_base_n = ((x_test_base[:] - norm_base["X_mean"]) / norm_base["X_std"]).astype(np.float32)

        att_b = StandardizedNet(4, 4, 64, norm_base["att_mean"], norm_base["att_std"], physics.ATT_MIN, physics.ATT_MAX).to(device)
        att_b.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device))
        att_b.eval()

        cbf_b = StandardizedNet(4, 5, 128, norm_base["cbf_mean"], norm_base["cbf_std"], physics.CBF_MIN, physics.CBF_MAX).to(device)
        cbf_b.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device))
        cbf_b.eval()

        p_cbf_base_dnn = cbf_b.predict_physical(X_test_base_n)
        p_att_base_dnn = att_b.predict_physical(X_test_base_n)

        elm_npz = np.load(os.path.join(ELM_DIR, "all_test_predictions.npz"))
        p_cbf_base_elm = elm_npz["ensemble_feature_elm_5_seeds_cbf_test"]
        p_att_base_elm = elm_npz["ensemble_feature_elm_5_seeds_att_test"]

    # 3. Load Optimized Acquisition Inputs & Models (Candidates 3 & 4)
    print("Loading Optimized Acquisition Inputs & Checkpoints...")
    norm_opt = np.load(os.path.join(CKPT_DIR, "normalization_stats_opt.npz"))
    X_mean_opt = norm_opt["X_mean"]
    X_std_opt = norm_opt["X_std"]
    cbf_mean = float(norm_opt["cbf_mean"])
    cbf_std = float(norm_opt["cbf_std"])
    att_mean = float(norm_opt["att_mean"])
    att_std = float(norm_opt["att_std"])

    x_test_opt = np.load(os.path.join(OPT_CACHE_DIR, "X_test_opt.npy"), mmap_mode='r')
    X_test_opt_n = ((x_test_opt[:] - X_mean_opt) / X_std_opt).astype(np.float32)

    att_opt_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    att_opt_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ATTnet_opt_4PLD.pth"), map_location=device))
    att_opt_net.eval()

    cbf_opt_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    cbf_opt_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "CBFnet_opt_4PLD.pth"), map_location=device))
    cbf_opt_net.eval()

    # Predict Candidate 3: Optimized 4-PLD DNN
    print("Predicting Candidate 3: Optimized 4-PLD DNN...")
    p_cbf_opt_dnn = cbf_opt_net.predict_physical(X_test_opt_n)
    p_att_opt_dnn = att_opt_net.predict_physical(X_test_opt_n)

    # Predict Candidate 4: Optimized 4-PLD Feature ELM
    print("Predicting Candidate 4: Optimized 4-PLD Feature ELM (5 seeds)...")
    x_tr_opt = np.load(os.path.join(OPT_CACHE_DIR, "X_train_opt.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    rng_elm = np.random.default_rng(2026)
    idx_elm = rng_elm.choice(1_000_000, size=100_000, replace=False)

    X_tr_sub_n = ((x_tr_opt[idx_elm] - X_mean_opt) / X_std_opt).astype(np.float32)
    Y_cbf_sub_std = ((y_tr_mmap[idx_elm, 0] - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_sub_std = ((y_tr_mmap[idx_elm, 1] - att_mean) / att_std).astype(np.float32)

    F_cbf_tr = cbf_opt_net.penultimate_features(X_tr_sub_n)
    F_att_tr = att_opt_net.penultimate_features(X_tr_sub_n)
    F_cbf_test = cbf_opt_net.penultimate_features(X_test_opt_n)
    F_att_test = att_opt_net.penultimate_features(X_test_opt_n)

    elm_seeds = [42, 43, 44, 45, 46]
    p_cbf_opt_elm = np.zeros(len(Y_cbf_test), dtype=np.float32)
    p_att_opt_elm = np.zeros(len(Y_att_test), dtype=np.float32)

    for s in elm_seeds:
        e_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_c.fit_chunked(F_cbf_tr, Y_cbf_sub_std)
        p_cbf_opt_elm += np.clip(e_c.predict(F_cbf_test) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)

        e_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_a.fit_chunked(F_att_tr, Y_att_sub_std)
        p_att_opt_elm += np.clip(e_a.predict(F_att_test) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

    p_cbf_opt_elm /= len(elm_seeds)
    p_att_opt_elm /= len(elm_seeds)

    # Save all test predictions
    np.savez(
        os.path.join(CKPT_DIR, "optimized_test_predictions.npz"),
        Y_cbf_test=Y_cbf_test, Y_att_test=Y_att_test, snr_test=snr_test,
        p_cbf_base_dnn=p_cbf_base_dnn, p_att_base_dnn=p_att_base_dnn,
        p_cbf_base_elm=p_cbf_base_elm, p_att_base_elm=p_att_base_elm,
        p_cbf_opt_dnn=p_cbf_opt_dnn, p_att_opt_dnn=p_att_opt_dnn,
        p_cbf_opt_elm=p_cbf_opt_elm, p_att_opt_elm=p_att_opt_elm
    )
    print(f"Saved all test predictions to: {os.path.join(CKPT_DIR, 'optimized_test_predictions.npz')}")

    # 4. Global Metric Calculation & Statistical Significance
    print("\nComputing Global Metrics & Statistical Tests (N=100,000)...")
    models = [
        ("Baseline A (DNN-G)", p_cbf_base_dnn, p_att_base_dnn),
        ("Baseline Feature ELM (5 seeds)", p_cbf_base_elm, p_att_base_elm),
        ("Optimized 4-PLD DNN", p_cbf_opt_dnn, p_att_opt_dnn),
        ("Optimized 4-PLD Feature ELM", p_cbf_opt_elm, p_att_opt_elm)
    ]

    base_cbf_m = calc_metrics(Y_cbf_test, p_cbf_base_dnn)
    base_att_m = calc_metrics(Y_att_test, p_att_base_dnn)

    se_base_c = (p_cbf_base_dnn - Y_cbf_test) ** 2
    se_base_a = (p_att_base_dnn - Y_att_test) ** 2
    ae_base_c = np.abs(p_cbf_base_dnn - Y_cbf_test)
    ae_base_a = np.abs(p_att_base_dnn - Y_att_test)

    rng_boot = np.random.default_rng(2026)
    sub_10k = rng_boot.choice(100_000, size=10_000, replace=False)
    n_boot = 1000

    global_rows = []
    for name, p_c, p_a in models:
        mc = calc_metrics(Y_cbf_test, p_c)
        ma = calc_metrics(Y_att_test, p_a)

        d_c_rmse = base_cbf_m["RMSE"] - mc["RMSE"]
        p_c_rmse = 100.0 * d_c_rmse / base_cbf_m["RMSE"]
        d_c_mae = base_cbf_m["MAE"] - mc["MAE"]
        p_c_mae = 100.0 * d_c_mae / base_cbf_m["MAE"]

        d_a_rmse = base_att_m["RMSE"] - ma["RMSE"]
        p_a_rmse = 100.0 * d_a_rmse / base_att_m["RMSE"]
        d_a_mae = base_att_m["MAE"] - ma["MAE"]
        p_a_mae = 100.0 * d_a_mae / base_att_m["MAE"]

        # Hypothesis testing against Baseline A
        if name == "Baseline A (DNN-G)":
            p_se_c, p_ae_c, p_w_c = 1.0, 1.0, 1.0
            p_se_a, p_ae_a, p_w_a = 1.0, 1.0, 1.0
            ci_c_r = (0.0, 0.0)
            ci_c_m = (0.0, 0.0)
            ci_a_r = (0.0, 0.0)
            ci_a_m = (0.0, 0.0)
        else:
            se_c = (p_c - Y_cbf_test) ** 2
            se_a = (p_a - Y_att_test) ** 2
            ae_c = np.abs(p_c - Y_cbf_test)
            ae_a = np.abs(p_a - Y_att_test)

            _, p_se_c = stats.ttest_rel(se_base_c, se_c)
            _, p_ae_c = stats.ttest_rel(ae_base_c, ae_c)
            _, p_w_c = stats.wilcoxon(ae_base_c[sub_10k], ae_c[sub_10k])

            _, p_se_a = stats.ttest_rel(se_base_a, se_a)
            _, p_ae_a = stats.ttest_rel(ae_base_a, ae_a)
            _, p_w_a = stats.wilcoxon(ae_base_a[sub_10k], ae_a[sub_10k])

            # Paired bootstrap CIs
            b_rmse_c = np.empty(n_boot)
            b_mae_c = np.empty(n_boot)
            b_rmse_a = np.empty(n_boot)
            b_mae_a = np.empty(n_boot)

            for b in range(n_boot):
                b_idx = rng_boot.integers(0, 100_000, size=100_000)
                b_rmse_c[b] = np.sqrt(np.mean(se_base_c[b_idx])) - np.sqrt(np.mean(se_c[b_idx]))
                b_mae_c[b] = np.mean(ae_base_c[b_idx]) - np.mean(ae_c[b_idx])
                b_rmse_a[b] = np.sqrt(np.mean(se_base_a[b_idx])) - np.sqrt(np.mean(se_a[b_idx]))
                b_mae_a[b] = np.mean(ae_base_a[b_idx]) - np.mean(ae_c[b_idx])

            ci_c_r = (float(np.percentile(b_rmse_c, 2.5)), float(np.percentile(b_rmse_c, 97.5)))
            ci_c_m = (float(np.percentile(b_mae_c, 2.5)), float(np.percentile(b_mae_c, 97.5)))
            ci_a_r = (float(np.percentile(b_rmse_a, 2.5)), float(np.percentile(b_rmse_a, 97.5)))
            ci_a_m = (float(np.percentile(b_mae_a, 2.5)), float(np.percentile(b_mae_a, 97.5)))

        global_rows.append({
            "Model": name,
            "CBF_RMSE": mc["RMSE"], "Delta_CBF_RMSE": d_c_rmse, "Gain_CBF_RMSE_%": p_c_rmse,
            "95%_CI_Delta_CBF_RMSE": f"[{ci_c_r[0]:.4f}, {ci_c_r[1]:.4f}]",
            "CBF_MAE": mc["MAE"], "Delta_CBF_MAE": d_c_mae, "Gain_CBF_MAE_%": p_c_mae,
            "95%_CI_Delta_CBF_MAE": f"[{ci_c_m[0]:.4f}, {ci_c_m[1]:.4f}]",
            "CBF_Bias": mc["Bias"], "CBF_CCC": mc["CCC"], "CBF_R2": mc["R2"],
            "p_val_MSE_CBF": p_se_c, "p_val_MAE_CBF": p_ae_c, "p_val_Wilcoxon_CBF": p_w_c,

            "ATT_RMSE": ma["RMSE"], "Delta_ATT_RMSE": d_a_rmse, "Gain_ATT_RMSE_%": p_a_rmse,
            "95%_CI_Delta_ATT_RMSE": f"[{ci_a_r[0]:.5f}, {ci_a_r[1]:.5f}]",
            "ATT_MAE": ma["MAE"], "Delta_ATT_MAE": d_a_mae, "Gain_ATT_MAE_%": p_a_mae,
            "95%_CI_Delta_ATT_MAE": f"[{ci_a_m[0]:.5f}, {ci_a_m[1]:.5f}]",
            "ATT_Bias": ma["Bias"], "ATT_CCC": ma["CCC"], "ATT_R2": ma["R2"],
            "p_val_MSE_ATT": p_se_a, "p_val_MAE_ATT": p_ae_a, "p_val_Wilcoxon_ATT": p_w_a
        })

    df_global = pd.DataFrame(global_rows)
    csv_global = os.path.join(TABLES_DIR, "optimized_vs_baseline_comparison.csv")
    df_global.to_csv(csv_global, index=False)
    print(f"\nSaved Global Comparison Table to: {csv_global}")

    print("\n--- Summary Performance on Locked Test Cohort (N=100,000) ---")
    cols_show = ["Model", "CBF_RMSE", "Gain_CBF_RMSE_%", "CBF_MAE", "Gain_CBF_MAE_%",
                 "ATT_RMSE", "Gain_ATT_RMSE_%", "ATT_MAE", "Gain_ATT_MAE_%", "ATT_CCC"]
    print(df_global[cols_show].to_string(index=False))

    # 5. ATT Interval Breakdown (Auditing Early Arrival Window [0.5, 1.0) s)
    print("\nComputing ATT Interval Breakdown (5 physiological intervals)...")
    intervals = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
    int_rows = []

    for lo, hi in intervals:
        mask_int = (Y_att_test >= lo) & (Y_att_test < hi) if hi < 3.0 else (Y_att_test >= lo) & (Y_att_test <= hi)
        n_int = int(np.sum(mask_int))
        int_lbl = f"[{lo:.1f}, {hi:.1f}]"

        for name, p_c, p_a in models:
            ma = calc_metrics(Y_att_test[mask_int], p_a[mask_int])
            mc = calc_metrics(Y_cbf_test[mask_int], p_c[mask_int])
            int_rows.append({
                "Interval": int_lbl,
                "Model": name,
                "Count": n_int,
                "ATT_RMSE": ma["RMSE"],
                "ATT_MAE": ma["MAE"],
                "ATT_Bias": ma["Bias"],
                "ATT_CCC": ma["CCC"],
                "CBF_RMSE": mc["RMSE"],
                "CBF_MAE": mc["MAE"]
            })

    df_int = pd.DataFrame(int_rows)
    csv_int = os.path.join(TABLES_DIR, "att_interval_comparison.csv")
    df_int.to_csv(csv_int, index=False)
    print(f"Saved ATT Interval Comparison Table to: {csv_int}")

    print("\n--- Early ATT Window [0.5, 1.0) s Performance (N=19,728) ---")
    df_early = df_int[df_int["Interval"] == "[0.5, 1.0]"]
    print(df_early[["Model", "ATT_RMSE", "ATT_MAE", "ATT_Bias", "ATT_CCC"]].to_string(index=False))

    # 6. SNR Breakdown Table
    snr_rows = []
    for s_val in physics.SNR_VALUES:
        mask_s = np.isinf(snr_test) if np.isinf(s_val) else (snr_test == s_val)
        n_s = int(np.sum(mask_s))
        s_lbl = "inf" if np.isinf(s_val) else f"{s_val:.1f}"

        for name, p_c, p_a in models:
            ma = calc_metrics(Y_att_test[mask_s], p_a[mask_s])
            mc = calc_metrics(Y_cbf_test[mask_s], p_c[mask_s])
            snr_rows.append({
                "SNR": s_lbl,
                "Model": name,
                "Count": n_s,
                "CBF_RMSE": mc["RMSE"], "CBF_MAE": mc["MAE"], "CBF_CCC": mc["CCC"],
                "ATT_RMSE": ma["RMSE"], "ATT_MAE": ma["MAE"], "ATT_CCC": ma["CCC"]
            })
    df_snr = pd.DataFrame(snr_rows)

    # 7. Diagnostic Visualizations
    print("\nGenerating Diagnostic Figures...")

    # Figure 1: Global Error Bars & Gains
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    models_short = [m.replace("Baseline A (DNN-G)", "Baseline DNN").replace("Baseline Feature ELM (5 seeds)", "Baseline ELM") for m in df_global["Model"]]
    y_pos = np.arange(len(models_short))

    axes[0].barh(y_pos - 0.2, df_global["CBF_RMSE"], height=0.4, label="CBF RMSE", color="#1f77b4", alpha=0.9)
    axes[0].barh(y_pos + 0.2, df_global["CBF_MAE"], height=0.4, label="CBF MAE", color="#aec7e8", alpha=0.9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models_short, fontsize=10)
    axes[0].set_xlabel("CBF Error [mL / (100g · min)]", fontsize=11)
    axes[0].set_title("Cerebral Blood Flow (CBF) Estimation Errors", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()

    axes[1].barh(y_pos - 0.2, df_global["ATT_RMSE"], height=0.4, label="ATT RMSE", color="#d62728", alpha=0.9)
    axes[1].barh(y_pos + 0.2, df_global["ATT_MAE"], height=0.4, label="ATT MAE", color="#ff9896", alpha=0.9)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(models_short, fontsize=10)
    axes[1].set_xlabel("ATT Error [seconds]", fontsize=11)
    axes[1].set_title("Arterial Transit Time (ATT) Estimation Errors", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()

    plt.tight_layout()
    p1_path = os.path.join(PLOTS_DIR, "optimized_vs_baseline_error_bars.png")
    plt.savefig(p1_path)
    plt.close()

    # Figure 2: ATT Interval Bias and RMSE Profiles
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    int_names = [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in intervals]

    for name, _, _ in models:
        sub = df_int[df_int["Model"] == name]
        rmse_vals = [sub[sub["Interval"] == lbl]["ATT_RMSE"].values[0] for lbl in int_names]
        bias_vals = [sub[sub["Interval"] == lbl]["ATT_Bias"].values[0] for lbl in int_names]

        axes[0].plot(int_names, rmse_vals, marker='o', lw=2.0, label=name)
        axes[1].plot(int_names, bias_vals, marker='s', lw=2.0, label=name)

    axes[0].set_xlabel("ATT Interval [seconds]", fontsize=11)
    axes[0].set_ylabel("ATT RMSE [seconds]", fontsize=11)
    axes[0].set_title("ATT RMSE Profile Across Intervals", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.0, color='black', ls='--', alpha=0.6)
    axes[1].set_xlabel("ATT Interval [seconds]", fontsize=11)
    axes[1].set_ylabel("ATT Bias (Predicted - True) [seconds]", fontsize=11)
    axes[1].set_title("ATT Bias Profile: Resolution of Early-Arrival Bottleneck", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    p2_path = os.path.join(PLOTS_DIR, "att_interval_bias_and_rmse.png")
    plt.savefig(p2_path)
    plt.close()

    # Figure 3: SNR Stratified Performance Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    snr_plot_levels = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0]

    for name, _, _ in models:
        sub = df_snr[df_snr["Model"] == name]
        cbf_r = [sub[sub["SNR"] == f"{s:.1f}"]["CBF_RMSE"].values[0] for s in snr_plot_levels]
        att_r = [sub[sub["SNR"] == f"{s:.1f}"]["ATT_RMSE"].values[0] for s in snr_plot_levels]

        axes[0].plot(snr_plot_levels, cbf_r, marker='o', lw=2.0, label=name)
        axes[1].plot(snr_plot_levels, att_r, marker='s', lw=2.0, label=name)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11)
    axes[0].set_ylabel("CBF RMSE [mL / (100g · min)]", fontsize=11)
    axes[0].set_title("CBF RMSE vs SNR", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11)
    axes[1].set_ylabel("ATT RMSE [seconds]", fontsize=11)
    axes[1].set_title("ATT RMSE vs SNR", fontsize=12, fontweight="bold")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    p3_path = os.path.join(PLOTS_DIR, "snr_stratified_gain_curves.png")
    plt.savefig(p3_path)
    plt.close()

    print(f"\nSaved all diagnostic figures to: {PLOTS_DIR}")
    print("=" * 80)
    print("PHASE 6 LOCKED EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

