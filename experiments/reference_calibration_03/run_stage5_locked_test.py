"""
run_stage5_locked_test.py
Stage 5: Final Evaluation of Selected Experiment 3 Models on the Locked Test Cohort (N=100,000).
Evaluates:
  1. Baseline A DNN (Experiment G DNN-G)
  2. Historical Feature ELM (5 seeds, H=512, alpha=1e-2, Rep A)
  3. Tuned Feature ELM (5 seeds, H=2048, optimal alphas, Rep A)
  4. Affine-Calibrated DNN (a * y + b)
  5. Affine-Calibrated Feature ELM (a * y + b)
  6. Optimal Shrinkage Blend (gamma* Feature ELM)
  7. Early-ATT Sigmoid-Damped ELM
Outputs:
  - final_test_metrics.csv
  - final_test_snr_stratified.csv
  - final_test_att_intervals.csv
  - final_test_predictions.npz
  - plots/final_test_evaluation.png
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
OUT_DIR = os.path.join(EXP_DIR, "reference_calibration_03")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.append(EXP_DIR)
sys.path.append(os.path.join(EXP_DIR, "elm_models"))

import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

def paired_bootstrap_ci(y_true, pred_base, pred_cand, B=1000, seed=42):
    """Compute 95% paired bootstrap confidence interval for RMSE difference (Base - Cand)."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    diffs = np.empty(B, dtype=np.float64)

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        yt_b = y_true[idx]
        pb_b = pred_base[idx]
        pc_b = pred_cand[idx]
        rmse_b = np.sqrt(np.mean((pb_b - yt_b) ** 2))
        rmse_c = np.sqrt(np.mean((pc_b - yt_b) ** 2))
        diffs[b] = rmse_b - rmse_c

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return ci_low, ci_high


def main():
    print("=" * 80)
    print("STAGE 5: LOCKED TEST SET EVALUATION (N=100,000)")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Normalization & Models
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    att_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device, weights_only=False))
    att_net.eval()

    cbf_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device, weights_only=False))
    cbf_net.eval()

    # 2. Calibration Parameters from Stage 4
    with open(os.path.join(OUT_DIR, "stage4_calibration_coefficients.json"), "r") as f:
        calib_params = json.load(f)

    # 3. Datasets
    N_tr = 100_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_te_mmap = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    y_te_mmap = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    snr_te = np.load(os.path.join(CACHE_DIR, "SNR_test.npy"))

    X_train_raw = np.array(x_tr_mmap[idx_tr], dtype=np.float32)
    Y_train_raw = np.array(y_tr_mmap[idx_tr], dtype=np.float32)
    X_train_dnn = (X_train_raw - X_mean) / X_std

    N_te = len(x_te_mmap)
    X_test_raw = np.array(x_te_mmap[:], dtype=np.float32)
    Y_test_raw = np.array(y_te_mmap[:], dtype=np.float32)
    X_test_dnn = (X_test_raw - X_mean) / X_std

    y_test_cbf = Y_test_raw[:, 0]
    y_test_att = Y_test_raw[:, 1]

    # 4. Extract Training Latents & Fit Feature ELMs
    print("\nExtracting Training Latents & Fitting Feature ELMs...")
    chunk_sz = 20_000
    F_tr_cbf = []
    F_tr_att = []
    for i in range(0, N_tr, chunk_sz):
        F_tr_cbf.append(cbf_net.penultimate_features(X_train_dnn[i:i+chunk_sz]))
        F_tr_att.append(att_net.penultimate_features(X_train_dnn[i:i+chunk_sz]))
    F_tr_cbf = np.vstack(F_tr_cbf)
    F_tr_att = np.vstack(F_tr_att)

    y_tr_cbf_std = (Y_train_raw[:, 0] - cbf_mean) / cbf_std
    y_tr_att_std = (Y_train_raw[:, 1] - att_mean) / att_std

    # Historical ELMs (H=512, alpha=1e-2, 5 seeds)
    hist_cbf_elms = []
    hist_att_elms = []
    for seed in [42, 101, 2024, 7, 99]:
        elm_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, activation="tanh", seed=seed)
        elm_c.fit_chunked(F_tr_cbf, y_tr_cbf_std)
        hist_cbf_elms.append(elm_c)

        elm_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, activation="tanh", seed=seed)
        elm_a.fit_chunked(F_tr_att, y_tr_att_std)
        hist_att_elms.append(elm_a)

    # Tuned ELMs (H=2048, alpha=10/1000, 5 seeds)
    tuned_cbf_elms = []
    tuned_att_elms = []
    for seed in [42, 101, 2024, 7, 99]:
        elm_c = GPUChunkedELM(n_hidden=2048, alpha=10.0, activation="tanh", seed=seed)
        elm_c.fit_chunked(F_tr_cbf, y_tr_cbf_std)
        tuned_cbf_elms.append(elm_c)

        elm_a = GPUChunkedELM(n_hidden=2048, alpha=1000.0, activation="tanh", seed=seed)
        elm_a.fit_chunked(F_tr_att, y_tr_att_std)
        tuned_att_elms.append(elm_a)

    del F_tr_cbf, F_tr_att
    torch.cuda.empty_cache()

    # 5. Generate Test Predictions
    print("\nGenerating Test Predictions across all models (N=100,000)...")
    # Base DNN
    pred_dnn_cbf = cbf_net.predict_physical(X_test_dnn)
    pred_dnn_att = att_net.predict_physical(X_test_dnn)

    # Extract Test Latents
    F_te_cbf = []
    F_te_att = []
    for i in range(0, N_te, chunk_sz):
        F_te_cbf.append(cbf_net.penultimate_features(X_test_dnn[i:i+chunk_sz]))
        F_te_att.append(att_net.penultimate_features(X_test_dnn[i:i+chunk_sz]))
    F_te_cbf = np.vstack(F_te_cbf)
    F_te_att = np.vstack(F_te_att)

    del cbf_net, att_net
    torch.cuda.empty_cache()

    # Historical ELM
    p_hist_cbf_list = [np.clip(elm.predict(F_te_cbf) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX) for elm in hist_cbf_elms]
    pred_hist_cbf = np.mean(p_hist_cbf_list, axis=0)
    p_hist_att_list = [np.clip(elm.predict(F_te_att) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX) for elm in hist_att_elms]
    pred_hist_att = np.mean(p_hist_att_list, axis=0)

    # Tuned ELM
    p_tuned_cbf_list = [np.clip(elm.predict(F_te_cbf) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX) for elm in tuned_cbf_elms]
    pred_tuned_cbf = np.mean(p_tuned_cbf_list, axis=0)
    p_tuned_att_list = [np.clip(elm.predict(F_te_att) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX) for elm in tuned_att_elms]
    pred_tuned_att = np.mean(p_tuned_att_list, axis=0)

    del F_te_cbf, F_te_att
    torch.cuda.empty_cache()

    # Affine-Calibrated DNN
    cbf_aff_dnn = calib_params["affine_cbf_dnn"]
    att_aff_dnn = calib_params["affine_att_dnn"]
    pred_aff_dnn_cbf = np.clip(cbf_aff_dnn["slope"] * pred_dnn_cbf + cbf_aff_dnn["intercept"], physics.CBF_MIN, physics.CBF_MAX)
    pred_aff_dnn_att = np.clip(att_aff_dnn["slope"] * pred_dnn_att + att_aff_dnn["intercept"], physics.ATT_MIN, physics.ATT_MAX)

    # Affine-Calibrated Feature ELM
    cbf_aff_elm = calib_params["affine_cbf_elm"]
    att_aff_elm = calib_params["affine_att_elm"]
    pred_aff_elm_cbf = np.clip(cbf_aff_elm["slope"] * pred_tuned_cbf + cbf_aff_elm["intercept"], physics.CBF_MIN, physics.CBF_MAX)
    pred_aff_elm_att = np.clip(att_aff_elm["slope"] * pred_tuned_att + att_aff_elm["intercept"], physics.ATT_MIN, physics.ATT_MAX)

    # Optimal Shrinkage Blend (gamma*)
    g_cbf = calib_params["optimal_shrinkage_gamma"]["CBF"]
    g_att = calib_params["optimal_shrinkage_gamma"]["ATT"]
    pred_blend_cbf = np.clip((1.0 - g_cbf) * pred_dnn_cbf + g_cbf * pred_tuned_cbf, physics.CBF_MIN, physics.CBF_MAX)
    pred_blend_att = np.clip((1.0 - g_att) * pred_dnn_att + g_att * pred_tuned_att, physics.ATT_MIN, physics.ATT_MAX)

    # Early-ATT Sigmoid-Damped ELM
    early_bias = calib_params["early_att_residual_correction"]["early_bias"]
    pred_damped_att = np.clip(pred_tuned_att - early_bias / (1.0 + np.exp((pred_tuned_att - 1.0) / 0.15)), physics.ATT_MIN, physics.ATT_MAX)

    # Save Prediction arrays
    np.savez_compressed(
        os.path.join(OUT_DIR, "final_test_predictions.npz"),
        pred_dnn_cbf=pred_dnn_cbf,
        pred_dnn_att=pred_dnn_att,
        pred_hist_cbf=pred_hist_cbf,
        pred_hist_att=pred_hist_att,
        pred_tuned_cbf=pred_tuned_cbf,
        pred_tuned_att=pred_tuned_att,
        pred_aff_dnn_cbf=pred_aff_dnn_cbf,
        pred_aff_dnn_att=pred_aff_dnn_att,
        pred_blend_cbf=pred_blend_cbf,
        pred_blend_att=pred_blend_att,
        y_test_cbf=y_test_cbf,
        y_test_att=y_test_att,
        snr_te=snr_te
    )
    print("Saved final_test_predictions.npz")

    # =========================================================================
    # COMPUTE OVERALL METRICS & PAIRED BOOTSTRAP CIS
    # =========================================================================
    models_dict = {
        "Baseline A (DNN-G)": (pred_dnn_cbf, pred_dnn_att, "StandardizedNet (Frozen 4-PLD)"),
        "Historical Feature ELM (5 seeds)": (pred_hist_cbf, pred_hist_att, "H=512, alpha=1e-2, tanh, Rep A (5 seeds)"),
        "Tuned Feature ELM (5 seeds)": (pred_tuned_cbf, pred_tuned_att, "H=2048, alpha=10/1000, Rep A (5 seeds)"),
        "Affine-Calibrated DNN": (pred_aff_dnn_cbf, pred_aff_dnn_att, "DNN with Stage 4 Affine Calibration"),
        "Affine-Calibrated Feature ELM": (pred_aff_elm_cbf, pred_aff_elm_att, "Feature ELM with Stage 4 Affine Calibration"),
        "Optimal Shrinkage Blend (gamma*)": (pred_blend_cbf, pred_blend_att, f"Shrinkage: {1-g_cbf:.2f}*DNN + {g_cbf:.2f}*ELM (CBF), {1-g_att:.2f}*DNN + {g_att:.2f}*ELM (ATT)"),
        "Early-ATT Sigmoid-Damped ELM": (pred_tuned_cbf, pred_damped_att, "Tuned ELM + Stage 4 Sigmoid Early Bias Subtraction")
    }

    base_rmse_cbf = calc_metrics(y_test_cbf, pred_dnn_cbf)["RMSE"]
    base_rmse_att = calc_metrics(y_test_att, pred_dnn_att)["RMSE"]
    hist_rmse_cbf = calc_metrics(y_test_cbf, pred_hist_cbf)["RMSE"]
    hist_rmse_att = calc_metrics(y_test_att, pred_hist_att)["RMSE"]

    early_mask = (y_test_att >= 0.5) & (y_test_att < 1.0)
    final_rows = []

    print("\nComputing Overall Metrics & 95% Bootstrap CIs...")
    for m_name, (p_cbf, p_att, m_cfg) in models_dict.items():
        mc = calc_metrics(y_test_cbf, p_cbf)
        ma = calc_metrics(y_test_att, p_att)
        early_ma = calc_metrics(y_test_att[early_mask], p_att[early_mask])

        gain_dnn_cbf = (base_rmse_cbf - mc["RMSE"]) / base_rmse_cbf * 100.0
        gain_hist_cbf = (hist_rmse_cbf - mc["RMSE"]) / hist_rmse_cbf * 100.0
        gain_dnn_att = (base_rmse_att - ma["RMSE"]) / base_rmse_att * 100.0
        gain_hist_att = (hist_rmse_att - ma["RMSE"]) / hist_rmse_att * 100.0

        if m_name == "Baseline A (DNN-G)":
            cbf_ci_low, cbf_ci_high = 0.0, 0.0
            att_ci_low, att_ci_high = 0.0, 0.0
        else:
            cbf_ci_low, cbf_ci_high = paired_bootstrap_ci(y_test_cbf, pred_dnn_cbf, p_cbf, B=1000)
            att_ci_low, att_ci_high = paired_bootstrap_ci(y_test_att, pred_dnn_att, p_att, B=1000)

        print(f"\n{m_name}:")
        print(f"  CBF: RMSE={mc['RMSE']:.4f}, MAE={mc['MAE']:.4f}, Bias={mc['Bias']:+.3f}, CCC={mc['CCC']:.4f}, Gain={gain_dnn_cbf:+.2f}%, 95% CI=[{cbf_ci_low:+.4f}, {cbf_ci_high:+.4f}]")
        print(f"  ATT: RMSE={ma['RMSE']:.4f}, MAE={ma['MAE']:.4f}, Bias={ma['Bias']:+.3f}, CCC={ma['CCC']:.4f}, Gain={gain_dnn_att:+.2f}%, 95% CI=[{att_ci_low:+.5f}, {att_ci_high:+.5f}]")
        print(f"  Early-ATT: RMSE={early_ma['RMSE']:.4f}, Bias={early_ma['Bias']:+.3f}, CCC={early_ma['CCC']:.4f}")

        final_rows.append({
            "Model": m_name,
            "Configuration": m_cfg,
            "CBF_RMSE": mc["RMSE"],
            "CBF_MAE": mc["MAE"],
            "CBF_Bias": mc["Bias"],
            "CBF_CCC": mc["CCC"],
            "CBF_R2": mc["R2"],
            "CBF_Gain_vs_DNN_%": gain_dnn_cbf,
            "CBF_Gain_vs_HistELM_%": gain_hist_cbf,
            "CBF_Delta_95CI_Low": cbf_ci_low,
            "CBF_Delta_95CI_High": cbf_ci_high,
            "ATT_RMSE": ma["RMSE"],
            "ATT_MAE": ma["MAE"],
            "ATT_Bias": ma["Bias"],
            "ATT_CCC": ma["CCC"],
            "ATT_R2": ma["R2"],
            "ATT_Gain_vs_DNN_%": gain_dnn_att,
            "ATT_Gain_vs_HistELM_%": gain_hist_att,
            "ATT_Delta_95CI_Low": att_ci_low,
            "ATT_Delta_95CI_High": att_ci_high,
            "Early_ATT_RMSE": early_ma["RMSE"],
            "Early_ATT_Bias": early_ma["Bias"],
            "Early_ATT_CCC": early_ma["CCC"]
        })

    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(os.path.join(OUT_DIR, "final_test_metrics.csv"), index=False)
    print("\nSaved final_test_metrics.csv")

    # =========================================================================
    # STRATIFIED SNR EVALUATION ACROSS ALL 12 LEVELS
    # =========================================================================
    print("\nComputing Stratified SNR Performance...")
    snr_levels = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, np.inf]
    snr_rows = []

    for snr_val in snr_levels:
        if np.isinf(snr_val):
            mask = np.isinf(snr_te)
            snr_str = "inf"
        else:
            mask = np.isclose(snr_te, snr_val)
            snr_str = str(int(snr_val) if snr_val.is_integer() else snr_val)

        for m_name, (p_cbf, p_att, _) in models_dict.items():
            mc = calc_metrics(y_test_cbf[mask], p_cbf[mask])
            ma = calc_metrics(y_test_att[mask], p_att[mask])
            snr_rows.append({
                "SNR": snr_str,
                "Model": m_name,
                "CBF_RMSE": mc["RMSE"],
                "CBF_MAE": mc["MAE"],
                "CBF_CCC": mc["CCC"],
                "ATT_RMSE": ma["RMSE"],
                "ATT_MAE": ma["MAE"],
                "ATT_CCC": ma["CCC"]
            })

    df_snr = pd.DataFrame(snr_rows)
    df_snr.to_csv(os.path.join(OUT_DIR, "final_test_snr_stratified.csv"), index=False)
    print("Saved final_test_snr_stratified.csv")

    # =========================================================================
    # STRATIFIED ATT CLINICAL WINDOW EVALUATION (5 WINDOWS)
    # =========================================================================
    print("\nComputing Stratified ATT Clinical Windows Performance...")
    att_windows = [
        ("[0.5, 1.0)", (y_test_att >= 0.5) & (y_test_att < 1.0)),
        ("[1.0, 1.5)", (y_test_att >= 1.0) & (y_test_att < 1.5)),
        ("[1.5, 2.0)", (y_test_att >= 1.5) & (y_test_att < 2.0)),
        ("[2.0, 2.5)", (y_test_att >= 2.0) & (y_test_att < 2.5)),
        ("[2.5, 3.0]", (y_test_att >= 2.5) & (y_test_att <= 3.0))
    ]

    win_rows = []
    for w_name, w_mask in att_windows:
        n_samples = int(np.sum(w_mask))
        for m_name, (p_cbf, p_att, _) in models_dict.items():
            mc = calc_metrics(y_test_cbf[w_mask], p_cbf[w_mask])
            ma = calc_metrics(y_test_att[w_mask], p_att[w_mask])
            win_rows.append({
                "ATT_Window": w_name,
                "Model": m_name,
                "Sample_Count": n_samples,
                "CBF_RMSE": mc["RMSE"],
                "CBF_MAE": mc["MAE"],
                "CBF_CCC": mc["CCC"],
                "ATT_RMSE": ma["RMSE"],
                "ATT_MAE": ma["MAE"],
                "ATT_Bias": ma["Bias"],
                "ATT_CCC": ma["CCC"]
            })

    df_win = pd.DataFrame(win_rows)
    df_win.to_csv(os.path.join(OUT_DIR, "final_test_att_intervals.csv"), index=False)
    print("Saved final_test_att_intervals.csv")

    # =========================================================================
    # GENERATE FINAL PUBLICATION-QUALITY PLOT
    # =========================================================================
    print("\nGenerating Final Evaluation Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    # 1. Overall Test RMSE Comparison
    ax1 = axes[0, 0]
    short_names = [
        "Base DNN",
        "Hist ELM",
        "Tuned ELM",
        "Affine DNN",
        "Affine ELM",
        "Shrinkage g*",
        "EarlyDamped"
    ]
    x_pos = np.arange(len(short_names))
    w = 0.35

    ax1.bar(x_pos - w/2, df_final["CBF_RMSE"], w, label="CBF RMSE (mL/100g/min)", color="#1f77b4", edgecolor="black")
    ax1.set_ylabel("CBF RMSE (mL/100g/min)", color="#1f77b4", fontweight="bold")
    ax1.set_ylim(3.5, 3.85)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9)
    ax1.set_title("Locked Test CBF & ATT RMSE (N=100,000)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_t = ax1.twinx()
    ax1_t.plot(x_pos, df_final["ATT_RMSE"], "o-", color="#d62728", linewidth=2, markersize=7, label="ATT RMSE (s)")
    ax1_t.set_ylabel("ATT RMSE (seconds)", color="#d62728", fontweight="bold")
    ax1_t.set_ylim(0.300, 0.325)
    ax1.legend(loc="upper left", fontsize=8)
    ax1_t.legend(loc="upper right", fontsize=8)

    # 2. ATT Interval RMSE
    ax2 = axes[0, 1]
    plot_models = [
        ("Baseline A (DNN-G)", "Baseline DNN", "#1f77b4", "-"),
        ("Historical Feature ELM (5 seeds)", "Historical ELM", "#2ca02c", "--"),
        ("Optimal Shrinkage Blend (gamma*)", "Shrinkage Blend", "#9467bd", "-."),
        ("Early-ATT Sigmoid-Damped ELM", "Early-Damped ELM", "#e377c2", ":")
    ]
    for m_key, m_lbl, col, ls in plot_models:
        sub = df_win[df_win["Model"] == m_key]
        ax2.plot(sub["ATT_Window"], sub["ATT_RMSE"], marker="s", label=m_lbl, color=col, linestyle=ls, linewidth=2)

    ax2.set_xlabel("Clinical ATT Transit Window (seconds)", fontweight="bold")
    ax2.set_ylabel("ATT RMSE (seconds)", fontweight="bold")
    ax2.set_title("ATT RMSE Across Clinical Transit Windows", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. CBF vs SNR
    ax3 = axes[1, 0]
    snr_order = ["5", "8", "10", "12", "15", "20", "30", "50", "100", "200", "500", "inf"]
    for m_key, m_lbl, col, ls in plot_models:
        sub = df_snr[df_snr["Model"] == m_key].copy()
        sub = sub.set_index("SNR").reindex(snr_order).reset_index()
        ax3.plot(sub["SNR"], sub["CBF_RMSE"], marker="o", label=m_lbl, color=col, linestyle=ls, linewidth=1.8)

    ax3.set_xlabel("Test SNR Level", fontweight="bold")
    ax3.set_ylabel("CBF RMSE (mL/100g/min)", fontweight="bold")
    ax3.set_title("CBF Performance Across All 12 SNR Levels", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. ATT vs SNR
    ax4 = axes[1, 1]
    for m_key, m_lbl, col, ls in plot_models:
        sub = df_snr[df_snr["Model"] == m_key].copy()
        sub = sub.set_index("SNR").reindex(snr_order).reset_index()
        ax4.plot(sub["SNR"], sub["ATT_RMSE"], marker="^", label=m_lbl, color=col, linestyle=ls, linewidth=1.8)

    ax4.set_xlabel("Test SNR Level", fontweight="bold")
    ax4.set_ylabel("ATT RMSE (seconds)", fontweight="bold")
    ax4.set_title("ATT Performance Across All 12 SNR Levels", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    eval_plot_path = os.path.join(PLOTS_DIR, "final_test_evaluation.png")
    plt.savefig(eval_plot_path, dpi=300)
    plt.close()
    print(f"Saved evaluation plot to: {eval_plot_path}")
    print("\nSTAGE 5 EVALUATION COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()

