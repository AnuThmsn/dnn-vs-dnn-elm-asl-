"""
generate_figures.py
Generates all 9 publication-quality figures for the paper reproduction:
1. CBF RMSE comparison across SNR (10, 15, 20)
2. ATT RMSE comparison across SNR (10, 15, 20)
3. CBF MAE comparison across SNR (10, 15, 20)
4. ATT MAE comparison across SNR (10, 15, 20)
5. CBF CCC comparison across SNR (10, 15, 20)
6. ATT CCC comparison across SNR (10, 15, 20)
7. CBF Ground Truth vs Predicted Scatter Plots (DNN-G vs H-DELM with identity line)
8. ATT Ground Truth vs Predicted Scatter Plots (DNN-G vs H-DELM with identity line)
9. Residual Error Distributions for DNN-G and H-DELM
All figures saved in PNG (300 DPI) and PDF format.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.4
})

def save_fig_both(fig, out_dir, base_name):
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{base_name}.png")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")

def plot_all_reproduction_figures(metrics_df, preds_dict, out_dir):
    """
    metrics_df columns: ['SNR', 'Model', 'Parameter', 'MSE', 'RMSE', 'MAE', 'Lin_CCC']
    preds_dict: dictionary with test predictions and ground truth.
    """
    os.makedirs(out_dir, exist_ok=True)

    snr_levels = [10.0, 15.0, 20.0]
    models = ["DNN-G", "H-DELM"]
    colors = {"DNN-G": "#1f77b4", "H-DELM": "#d62728"}
    markers = {"DNN-G": "o", "H-DELM": "s"}

    # 1. CBF RMSE across SNR
    fig, ax = plt.subplots(figsize=(6, 4.5))
    df_cbf = metrics_df[metrics_df["Parameter"] == "CBF"]
    for m in models:
        sub = df_cbf[df_cbf["Model"] == m].sort_values("SNR")
        ax.plot(sub["SNR"], sub["RMSE"], marker=markers[m], color=colors[m], label=m, markersize=7)
    ax.set_title("CBF RMSE across SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE (mL/100g/min)")
    ax.set_xticks(snr_levels)
    ax.legend(frameon=True)
    save_fig_both(fig, out_dir, "01_cbf_rmse_vs_snr")

    # 2. ATT RMSE across SNR
    fig, ax = plt.subplots(figsize=(6, 4.5))
    df_att = metrics_df[metrics_df["Parameter"] == "ATT"]
    for m in models:
        sub = df_att[df_att["Model"] == m].sort_values("SNR")
        ax.plot(sub["SNR"], sub["RMSE"], marker=markers[m], color=colors[m], label=m, markersize=7)
    ax.set_title("ATT RMSE across SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE (seconds)")
    ax.set_xticks(snr_levels)
    ax.legend(frameon=True)
    save_fig_both(fig, out_dir, "02_att_rmse_vs_snr")

    # 3. CBF MAE across SNR
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in models:
        sub = df_cbf[df_cbf["Model"] == m].sort_values("SNR")
        ax.plot(sub["SNR"], sub["MAE"], marker=markers[m], color=colors[m], label=m, markersize=7)
    ax.set_title("CBF MAE across SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MAE (mL/100g/min)")
    ax.set_xticks(snr_levels)
    ax.legend(frameon=True)
    save_fig_both(fig, out_dir, "03_cbf_mae_vs_snr")

    # 4. ATT MAE across SNR
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in models:
        sub = df_att[df_att["Model"] == m].sort_values("SNR")
        ax.plot(sub["SNR"], sub["MAE"], marker=markers[m], color=colors[m], label=m, markersize=7)
    ax.set_title("ATT MAE across SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MAE (seconds)")
    ax.set_xticks(snr_levels)
    ax.legend(frameon=True)
    save_fig_both(fig, out_dir, "04_att_mae_vs_snr")

    # 5. CBF CCC across SNR
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in models:
        sub = df_cbf[df_cbf["Model"] == m].sort_values("SNR")
        ax.plot(sub["SNR"], sub["Lin_CCC"], marker=markers[m], color=colors[m], label=m, markersize=7)
    ax.set_title("CBF Lin's CCC across SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Lin's Concordance Correlation (CCC)")
    ax.set_xticks(snr_levels)
    ax.legend(frameon=True)
    save_fig_both(fig, out_dir, "05_cbf_ccc_vs_snr")

    # 6. ATT CCC across SNR
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in models:
        sub = df_att[df_att["Model"] == m].sort_values("SNR")
        ax.plot(sub["SNR"], sub["Lin_CCC"], marker=markers[m], color=colors[m], label=m, markersize=7)
    ax.set_title("ATT Lin's CCC across SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Lin's Concordance Correlation (CCC)")
    ax.set_xticks(snr_levels)
    ax.legend(frameon=True)
    save_fig_both(fig, out_dir, "06_att_ccc_vs_snr")

    # 7. CBF Scatter Plots (Ground Truth vs Predicted)
    cbf_true = preds_dict["cbf_true"]
    cbf_dnn = preds_dict["cbf_dnn"]
    cbf_hdelm = preds_dict["cbf_hdelm"]

    # Subsample for clear visualization
    sub_idx = np.random.default_rng(42).choice(len(cbf_true), size=min(4000, len(cbf_true)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, sharex=True)
    # DNN-G
    axes[0].scatter(cbf_true[sub_idx], cbf_dnn[sub_idx], alpha=0.25, s=6, color="#1f77b4", rasterized=True)
    axes[0].plot([0, 100], [0, 100], "k--", lw=1.5, label="Identity (1:1)")
    axes[0].set_title("DNN-G Baseline: CBF Estimation")
    axes[0].set_xlabel("Ground Truth CBF (mL/100g/min)")
    axes[0].set_ylabel("Predicted CBF (mL/100g/min)")
    axes[0].set_xlim([0, 100])
    axes[0].set_ylim([0, 100])
    axes[0].legend(loc="upper left")

    # H-DELM
    axes[1].scatter(cbf_true[sub_idx], cbf_hdelm[sub_idx], alpha=0.25, s=6, color="#d62728", rasterized=True)
    axes[1].plot([0, 100], [0, 100], "k--", lw=1.5, label="Identity (1:1)")
    axes[1].set_title("Proposed Residual H-DELM: CBF Estimation")
    axes[1].set_xlabel("Ground Truth CBF (mL/100g/min)")
    axes[1].set_xlim([0, 100])
    axes[1].legend(loc="upper left")

    save_fig_both(fig, out_dir, "07_cbf_prediction_scatter")

    # 8. ATT Scatter Plots (Ground Truth vs Predicted)
    att_true = preds_dict["att_true"]
    att_dnn = preds_dict["att_dnn"]
    att_hdelm = preds_dict["att_hdelm"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, sharex=True)
    # DNN-G
    axes[0].scatter(att_true[sub_idx], att_dnn[sub_idx], alpha=0.25, s=6, color="#1f77b4", rasterized=True)
    axes[0].plot([0.5, 3.0], [0.5, 3.0], "k--", lw=1.5, label="Identity (1:1)")
    axes[0].set_title("DNN-G Baseline: ATT Estimation")
    axes[0].set_xlabel("Ground Truth ATT (seconds)")
    axes[0].set_ylabel("Predicted ATT (seconds)")
    axes[0].set_xlim([0.5, 3.0])
    axes[0].set_ylim([0.5, 3.0])
    axes[0].legend(loc="upper left")

    # H-DELM
    axes[1].scatter(att_true[sub_idx], att_hdelm[sub_idx], alpha=0.25, s=6, color="#d62728", rasterized=True)
    axes[1].plot([0.5, 3.0], [0.5, 3.0], "k--", lw=1.5, label="Identity (1:1)")
    axes[1].set_title("Proposed Residual H-DELM: ATT Estimation")
    axes[1].set_xlabel("Ground Truth ATT (seconds)")
    axes[1].set_xlim([0.5, 3.0])
    axes[1].legend(loc="upper left")

    save_fig_both(fig, out_dir, "08_att_prediction_scatter")

    # 9. Residual Error Distributions
    res_cbf_dnn = cbf_dnn - cbf_true
    res_cbf_hdelm = cbf_hdelm - cbf_true
    res_att_dnn = att_dnn - att_true
    res_att_hdelm = att_hdelm - att_true

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # CBF residuals
    axes[0].hist(res_cbf_dnn, bins=80, range=[-15, 15], density=True, alpha=0.5, color="#1f77b4", label="DNN-G")
    axes[0].hist(res_cbf_hdelm, bins=80, range=[-15, 15], density=True, alpha=0.5, color="#d62728", label="H-DELM")
    axes[0].axvline(0, color="k", linestyle="--", lw=1)
    axes[0].set_title("CBF Residual Error Distribution ($y_{pred} - y_{true}$)")
    axes[0].set_xlabel("CBF Error (mL/100g/min)")
    axes[0].set_ylabel("Probability Density")
    axes[0].legend()

    # ATT residuals
    axes[1].hist(res_att_dnn, bins=80, range=[-1.0, 1.0], density=True, alpha=0.5, color="#1f77b4", label="DNN-G")
    axes[1].hist(res_att_hdelm, bins=80, range=[-1.0, 1.0], density=True, alpha=0.5, color="#d62728", label="H-DELM")
    axes[1].axvline(0, color="k", linestyle="--", lw=1)
    axes[1].set_title("ATT Residual Error Distribution ($y_{pred} - y_{true}$)")
    axes[1].set_xlabel("ATT Error (seconds)")
    axes[1].set_ylabel("Probability Density")
    axes[1].legend()

    save_fig_both(fig, out_dir, "09_residual_error_distributions")
    print(f"All 9 publication figures generated and saved successfully in {out_dir}.")

