"""
run_stage4_diagnostics.py
Stage 4: Validation Diagnostics, SNR Stratification, ATT Interval Analysis,
Resource Benchmarking, and Candidate Selection for Experiment 4.
Generates comprehensive diagnostic tables and publication-quality plots.
"""

import os
import sys
import gc
import time
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Environment and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments", "elm_models"))

import memory_config
from physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, SNR_VALUES
)
from train_baseline_A import StandardizedNet, calc_metrics
from run_stage3_nonlinear_readouts import MultiLayerExtractor, ReadoutModel

OUT_DIR = os.path.join(BASE_DIR, "experiments", "multilayer_elm_04")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
CACHE_DIR = os.path.join(BASE_DIR, "experiments", "cache")
os.makedirs(PLOT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stage 4 Diagnostics] Device: {device}")

def main():
    print("=" * 80)
    print("STAGE 4: VALIDATION DIAGNOSTICS & CANDIDATE SELECTION")
    print("=" * 80)

    # 1. Load Data
    norm_path = os.path.join(BASE_DIR, "experiments", "baseline_A", "normalization_stats.npz")
    norm_data = np.load(norm_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    y_cbf_mean = float(norm_data["cbf_mean"])
    y_cbf_std = float(norm_data["cbf_std"])
    y_att_mean = float(norm_data["att_mean"])
    y_att_std = float(norm_data["att_std"])

    cbf_net = StandardizedNet(4, 5, 128, y_cbf_mean, y_cbf_std, CBF_MIN, CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "experiments", "baseline_A", "CBFnet_G_baselineA.pth"), map_location=device))
    cbf_net.eval()

    att_net = StandardizedNet(4, 4, 64, y_att_mean, y_att_std, ATT_MIN, ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "experiments", "baseline_A", "ATTnet_G_baselineA.pth"), map_location=device))
    att_net.eval()

    cbf_extractor = MultiLayerExtractor(cbf_net, is_cbf=True)
    att_extractor = MultiLayerExtractor(att_net, is_cbf=False)

    X_tr_raw = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    Y_tr_raw = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    X_val_raw = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    Y_val_raw = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')
    SNR_val = np.load(os.path.join(CACHE_DIR, "SNR_val.npy"), mmap_mode='r') if os.path.exists(os.path.join(CACHE_DIR, "SNR_val.npy")) else None

    # If SNR_val doesn't exist in cache, check if test snrs or generate / tile
    if SNR_val is None:
        # Tile SNR values uniformly as done in standard dataset generation
        n_val = X_val_raw.shape[0]
        n_per_snr = n_val // len(SNR_VALUES)
        SNR_val = np.repeat(SNR_VALUES, n_per_snr)[:n_val]

    X_tr_std = (X_tr_raw[:] - X_mean) / X_std
    X_val_std = (X_val_raw[:] - X_mean) / X_std

    Y_cbf_tr_std = (Y_tr_raw[:, 0] - y_cbf_mean) / y_cbf_std
    Y_att_tr_std = (Y_tr_raw[:, 1] - y_att_mean) / y_att_std

    Y_cbf_val = Y_val_raw[:, 0].astype(np.float64)
    Y_att_val = Y_val_raw[:, 1].astype(np.float64)

    # 2. Train Selected Candidates for Deep Diagnostics
    # Model 1: Base DNN
    t0 = time.time()
    pred_cbf_dnn = cbf_net.predict_physical(X_val_std)
    pred_att_dnn = att_net.predict_physical(X_val_std)
    t_dnn_infer = time.time() - t0

    # Model 2: Historical Feature ELM (Rep A, Tanh 512, alpha=1e-2)
    t0 = time.time()
    m_hist_cbf = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_hist_cbf.accumulate_HtH_HtY(cbf_extractor, "Rep_A", X_tr_std, Y_cbf_tr_std)
    beta = m_hist_cbf.solve_alpha(HtH, HtY, 1e-2)
    t_hist_cbf_train = time.time() - t0
    t0 = time.time()
    pred_cbf_hist = np.clip(m_hist_cbf.predict_from_beta(cbf_extractor, "Rep_A", X_val_std, beta) * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
    t_hist_cbf_infer = time.time() - t0
    del HtH, HtY, beta

    t0 = time.time()
    m_hist_att = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_hist_att.accumulate_HtH_HtY(att_extractor, "Rep_A", X_tr_std, Y_att_tr_std)
    beta = m_hist_att.solve_alpha(HtH, HtY, 1e-2)
    t_hist_att_train = time.time() - t0
    t0 = time.time()
    pred_att_hist = np.clip(m_hist_att.predict_from_beta(att_extractor, "Rep_A", X_val_std, beta) * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
    t_hist_att_infer = time.time() - t0
    del HtH, HtY, beta

    # Model 3: Best Penultimate Candidate (Rep A, Tanh 1024, alpha=1.0 for CBF; Rep A, Tanh 512, alpha=10.0 for ATT)
    t0 = time.time()
    m_penult_cbf = ReadoutModel("elm", "tanh", 1024, seed=42)
    HtH, HtY = m_penult_cbf.accumulate_HtH_HtY(cbf_extractor, "Rep_A", X_tr_std, Y_cbf_tr_std)
    beta = m_penult_cbf.solve_alpha(HtH, HtY, 1.0)
    t_penult_cbf_train = time.time() - t0
    t0 = time.time()
    pred_cbf_penult = np.clip(m_penult_cbf.predict_from_beta(cbf_extractor, "Rep_A", X_val_std, beta) * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
    t_penult_cbf_infer = time.time() - t0
    del HtH, HtY, beta

    t0 = time.time()
    m_penult_att = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_penult_att.accumulate_HtH_HtY(att_extractor, "Rep_A", X_tr_std, Y_att_tr_std)
    beta = m_penult_att.solve_alpha(HtH, HtY, 10.0)
    t_penult_att_train = time.time() - t0
    t0 = time.time()
    pred_att_penult = np.clip(m_penult_att.predict_from_beta(att_extractor, "Rep_A", X_val_std, beta) * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
    t_penult_att_infer = time.time() - t0
    del HtH, HtY, beta

    # Model 4: Best Multi-Layer Candidate (Rep C [h1, h5], Tanh 512 for CBF; Rep C [h1, h4], Tanh 512 for ATT)
    t0 = time.time()
    m_multi_cbf = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_multi_cbf.accumulate_HtH_HtY(cbf_extractor, "Rep_C", X_tr_std, Y_cbf_tr_std)
    beta = m_multi_cbf.solve_alpha(HtH, HtY, 0.1)
    t_multi_cbf_train = time.time() - t0
    t0 = time.time()
    pred_cbf_multi = np.clip(m_multi_cbf.predict_from_beta(cbf_extractor, "Rep_C", X_val_std, beta) * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
    t_multi_cbf_infer = time.time() - t0
    del HtH, HtY, beta

    t0 = time.time()
    m_multi_att = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_multi_att.accumulate_HtH_HtY(att_extractor, "Rep_C", X_tr_std, Y_att_tr_std)
    beta = m_multi_att.solve_alpha(HtH, HtY, 10.0)
    t_multi_att_train = time.time() - t0
    t0 = time.time()
    pred_att_multi = np.clip(m_multi_att.predict_from_beta(att_extractor, "Rep_C", X_val_std, beta) * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
    t_multi_att_infer = time.time() - t0
    del HtH, HtY, beta

    # Model 5: Compact Linear Ridge Readout (Benchmark)
    t0 = time.time()
    m_lin_cbf = ReadoutModel("linear", seed=42)
    HtH, HtY = m_lin_cbf.accumulate_HtH_HtY(cbf_extractor, "Rep_A", X_tr_std, Y_cbf_tr_std)
    beta = m_lin_cbf.solve_alpha(HtH, HtY, 10.0)
    t_lin_cbf_train = time.time() - t0
    t0 = time.time()
    pred_cbf_linear = np.clip(m_lin_cbf.predict_from_beta(cbf_extractor, "Rep_A", X_val_std, beta) * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
    t_lin_cbf_infer = time.time() - t0
    del HtH, HtY, beta

    t0 = time.time()
    m_lin_att = ReadoutModel("linear", seed=42)
    HtH, HtY = m_lin_att.accumulate_HtH_HtY(att_extractor, "Rep_A", X_tr_std, Y_att_tr_std)
    beta = m_lin_att.solve_alpha(HtH, HtY, 10.0)
    t_lin_att_train = time.time() - t0
    t0 = time.time()
    pred_att_linear = np.clip(m_lin_att.predict_from_beta(att_extractor, "Rep_A", X_val_std, beta) * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
    t_lin_att_infer = time.time() - t0
    del HtH, HtY, beta

    # Compile Overall Model Comparison Table
    models_dict = {
        "Baseline A (DNN-G)": (pred_cbf_dnn, pred_att_dnn, 0.0, 0.0, t_dnn_infer, t_dnn_infer),
        "Historical Feature ELM (Rep A, 512, a=0.01)": (pred_cbf_hist, pred_att_hist, t_hist_cbf_train, t_hist_att_train, t_hist_cbf_infer, t_hist_att_infer),
        "Penultimate Candidate (Rep A, Tuned)": (pred_cbf_penult, pred_att_penult, t_penult_cbf_train, t_penult_att_train, t_penult_cbf_infer, t_penult_att_infer),
        "Multi-Layer Candidate (Rep C, Tuned)": (pred_cbf_multi, pred_att_multi, t_multi_cbf_train, t_multi_att_train, t_multi_cbf_infer, t_multi_att_infer),
        "Compact Linear Ridge (Rep A, No ELM)": (pred_cbf_linear, pred_att_linear, t_lin_cbf_train, t_lin_att_train, t_lin_cbf_infer, t_lin_att_infer)
    }

    overall_rows = []
    for name, (pcbf, patt, tr_cbf, tr_att, inf_cbf, inf_att) in models_dict.items():
        m_cbf = calc_metrics(Y_cbf_val, pcbf)
        m_att = calc_metrics(Y_att_val, patt)
        
        # Early ATT mask [0.5, 1.0)
        early_mask = (Y_att_val >= 0.5) & (Y_att_val < 1.0)
        m_early_att = calc_metrics(Y_att_val[early_mask], patt[early_mask])

        cbf_gain = (models_dict["Baseline A (DNN-G)"][0] is not pcbf)
        pct_cbf = (models_dict["Baseline A (DNN-G)"][0] is not None)
        cbf_dnn_rmse = calc_metrics(Y_cbf_val, models_dict["Baseline A (DNN-G)"][0])["RMSE"]
        att_dnn_rmse = calc_metrics(Y_att_val, models_dict["Baseline A (DNN-G)"][1])["RMSE"]

        cbf_imp = (cbf_dnn_rmse - m_cbf["RMSE"]) / cbf_dnn_rmse * 100.0
        att_imp = (att_dnn_rmse - m_att["RMSE"]) / att_dnn_rmse * 100.0

        overall_rows.append({
            "Model": name,
            "CBF_RMSE": round(m_cbf["RMSE"], 4),
            "CBF_MAE": round(m_cbf["MAE"], 4),
            "CBF_CCC": round(m_cbf["CCC"], 4),
            "CBF_R2": round(m_cbf["R2"], 4),
            "CBF_Bias": round(m_cbf["Bias"], 4),
            "CBF_Imp_%": round(cbf_imp, 2),
            "ATT_RMSE": round(m_att["RMSE"], 4),
            "ATT_MAE": round(m_att["MAE"], 4),
            "ATT_CCC": round(m_att["CCC"], 4),
            "ATT_R2": round(m_att["R2"], 4),
            "ATT_Bias": round(m_att["Bias"], 4),
            "Early_ATT_RMSE": round(m_early_att["RMSE"], 4),
            "ATT_Imp_%": round(att_imp, 2),
            "Train_Time_s": round(tr_cbf + tr_att, 2),
            "Infer_Time_s": round(inf_cbf + inf_att, 2)
        })

    val_comp_df = pd.DataFrame(overall_rows)
    val_comp_csv = os.path.join(OUT_DIR, "stage4_validation_comparison.csv")
    val_comp_df.to_csv(val_comp_csv, index=False)
    print("\n--- Validation Candidate Comparison Table ---")
    print(val_comp_df[["Model", "CBF_RMSE", "CBF_Imp_%", "ATT_RMSE", "Early_ATT_RMSE", "ATT_Imp_%", "Train_Time_s"]].to_string())

    # 3. SNR Stratification Diagnostics
    print("\nComputing SNR-Stratified Metrics across 12 levels...")
    snr_rows = []
    unique_snrs = np.unique(SNR_val)
    # Sort SNR with inf at end
    unique_snrs = sorted(unique_snrs, key=lambda x: (np.isinf(x), x))

    for snr in unique_snrs:
        mask = (SNR_val == snr) if not np.isinf(snr) else np.isinf(SNR_val)
        snr_label = f"{snr:.1f} dB" if not np.isinf(snr) else "Inf dB"

        for name, (pcbf, patt, _, _, _, _) in models_dict.items():
            m_cbf = calc_metrics(Y_cbf_val[mask], pcbf[mask])
            m_att = calc_metrics(Y_att_val[mask], patt[mask])
            snr_rows.append({
                "SNR": snr_label,
                "Model": name,
                "CBF_RMSE": round(m_cbf["RMSE"], 4),
                "CBF_MAE": round(m_cbf["MAE"], 4),
                "CBF_CCC": round(m_cbf["CCC"], 4),
                "ATT_RMSE": round(m_att["RMSE"], 4),
                "ATT_MAE": round(m_att["MAE"], 4),
                "ATT_CCC": round(m_att["CCC"], 4)
            })

    snr_df = pd.DataFrame(snr_rows)
    snr_csv = os.path.join(OUT_DIR, "stage4_snr_stratified.csv")
    snr_df.to_csv(snr_csv, index=False)

    # 4. ATT Physiological Interval Diagnostics
    print("\nComputing ATT Interval Breakdown...")
    intervals = [
        ("[0.5, 1.0) s (Early)", 0.5, 1.0),
        ("[1.0, 1.5) s (Normal GM)", 1.0, 1.5),
        ("[1.5, 2.0) s (Intermediate)", 1.5, 2.0),
        ("[2.0, 2.5) s (Delayed)", 2.0, 2.5),
        ("[2.5, 3.0] s (Late)", 2.5, 3.001)
    ]
    att_interval_rows = []
    for label, low, high in intervals:
        mask = (Y_att_val >= low) & (Y_att_val < high)
        n_samples = int(np.sum(mask))

        for name, (pcbf, patt, _, _, _, _) in models_dict.items():
            m_att = calc_metrics(Y_att_val[mask], patt[mask])
            att_interval_rows.append({
                "Interval": label,
                "N_Samples": n_samples,
                "Model": name,
                "ATT_RMSE": round(m_att["RMSE"], 4),
                "ATT_MAE": round(m_att["MAE"], 4),
                "ATT_Bias": round(m_att["Bias"], 4),
                "ATT_CCC": round(m_att["CCC"], 4)
            })

    att_int_df = pd.DataFrame(att_interval_rows)
    att_int_csv = os.path.join(OUT_DIR, "stage4_att_intervals.csv")
    att_int_df.to_csv(att_int_csv, index=False)

    # 5. Publication-Quality Plots
    print("\nGenerating Diagnostic Plots in experiments/multilayer_elm_04/plots/...")

    # Plot 1: Representation Comparison (Stage 2)
    rep_csv = os.path.join(OUT_DIR, "stage2_representation_comparison.csv")
    if os.path.exists(rep_csv):
        rep_df = pd.read_csv(rep_csv)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        cbf_reps = rep_df[rep_df["Target"] == "CBF"]
        axes[0].bar(cbf_reps["Representation"], cbf_reps["Val_RMSE"], color="steelblue", edgecolor="black", alpha=0.85)
        axes[0].axhline(y=3.7411, color="crimson", linestyle="--", linewidth=1.5, label="Baseline DNN (3.7411)")
        axes[0].set_title("CBF Validation RMSE by Representation", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("RMSE (mL / 100g / min)", fontsize=11)
        axes[0].set_ylim(3.68, 3.76)
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right")
        for i, v in enumerate(cbf_reps["Val_RMSE"]):
            axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

        att_reps = rep_df[rep_df["Target"] == "ATT"]
        axes[1].bar(att_reps["Representation"], att_reps["Val_RMSE"], color="coral", edgecolor="black", alpha=0.85)
        axes[1].axhline(y=0.3042, color="crimson", linestyle="--", linewidth=1.5, label="Baseline DNN (0.3042)")
        axes[1].set_title("ATT Validation RMSE by Representation", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("RMSE (seconds)", fontsize=11)
        axes[1].set_ylim(0.302, 0.306)
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right")
        for i, v in enumerate(att_reps["Val_RMSE"]):
            axes[1].text(i, v + 0.0001, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "stage2_representation_comparison.png"), dpi=300)
        plt.close()

    # Plot 2: Performance vs SNR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_snrs_labels = [s for s in unique_snrs if not np.isinf(s)]
    plot_snrs_str = [f"{s:.1f} dB" for s in plot_snrs_labels]
    
    models_to_plot = ["Baseline A (DNN-G)", "Historical Feature ELM (Rep A, 512, a=0.01)", "Penultimate Candidate (Rep A, Tuned)", "Multi-Layer Candidate (Rep C, Tuned)"]
    markers = ["o", "s", "^", "D"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for idx, m_name in enumerate(models_to_plot):
        sub_df = snr_df[(snr_df["Model"] == m_name) & (snr_df["SNR"].isin(plot_snrs_str))]
        sub_cbf = [sub_df[sub_df["SNR"] == s]["CBF_RMSE"].values[0] for s in plot_snrs_str]
        sub_att = [sub_df[sub_df["SNR"] == s]["ATT_RMSE"].values[0] for s in plot_snrs_str]

        axes[0].plot(plot_snrs_labels, sub_cbf, marker=markers[idx], color=colors[idx], label=m_name, linewidth=1.8, markersize=6)
        axes[1].plot(plot_snrs_labels, sub_att, marker=markers[idx], color=colors[idx], label=m_name, linewidth=1.8, markersize=6)

    axes[0].set_title("CBF RMSE vs Acquisition SNR", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=11)
    axes[0].set_ylabel("CBF RMSE (mL / 100g / min)", fontsize=11)
    axes[0].set_xscale("log")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    axes[1].set_title("ATT RMSE vs Acquisition SNR", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=11)
    axes[1].set_ylabel("ATT RMSE (seconds)", fontsize=11)
    axes[1].set_xscale("log")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "stage4_snr_performance.png"), dpi=300)
    plt.close()

    # Plot 3: ATT Interval Breakdown
    fig, ax = plt.subplots(figsize=(9, 5))
    int_labels = [row[0] for row in intervals]
    x = np.arange(len(int_labels))
    width = 0.2

    for i, m_name in enumerate(models_to_plot):
        sub_int = att_int_df[att_int_df["Model"] == m_name]
        rmses = [sub_int[sub_int["Interval"] == l]["ATT_RMSE"].values[0] for l in int_labels]
        ax.bar(x + (i - 1.5) * width, rmses, width, label=m_name, color=colors[i], edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(int_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("ATT RMSE (seconds)", fontsize=11)
    ax.set_title("ATT Estimation Accuracy across Bolus Arrival Intervals", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "stage4_att_intervals.png"), dpi=300)
    plt.close()

    print("\nStage 4 Diagnostics complete. All files saved.")

if __name__ == "__main__":
    main()

