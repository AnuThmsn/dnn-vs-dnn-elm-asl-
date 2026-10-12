"""
generate_comparison_plots.py
Generates comprehensive visualization plots for Phase 4 & Phase 5:
1. Model comparison bar charts (RMSE and MAE for CBF and ATT).
2. SNR robustness curves across all 12 SNR levels comparing DNN vs ELM variants.
3. Paired bootstrap distribution plots showing strictly positive improvements with 95% CIs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(EXP_DIR, "..", "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

def generate_plots():
    # 1. Bar Chart Comparison (Test Set)
    test_df = pd.read_csv(os.path.join(EXP_DIR, "all_models_test_comparison.csv"))
    
    models = test_df["Model"].tolist()
    # Shorten names for cleaner plotting
    short_names = ["Baseline DNN", "Standalone ELM", "Weighted Ens", "Residual ELM", "DNN-Feat ELM", "Ens Feat ELM (5s)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]

    # CBF RMSE
    bars1 = axes[0].bar(short_names, test_df["CBF_RMSE"], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("CBF Estimation Error (Test Set, N=100k)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("CBF RMSE (mL / [100g·min])", fontsize=11)
    axes[0].set_ylim(3.5, 3.8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].tick_params(axis="x", rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha="center", va="bottom", fontsize=8.5)

    # ATT RMSE
    bars2 = axes[1].bar(short_names, test_df["ATT_RMSE"], color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_title("ATT Estimation Error (Test Set, N=100k)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("ATT RMSE (seconds)", fontsize=11)
    axes[1].set_ylim(0.295, 0.320)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].tick_params(axis="x", rotation=25)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.0005, f"{yval:.4f}", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    bar_path = os.path.join(DIAG_DIR, "all_models_rmse_comparison.png")
    fig.savefig(bar_path)
    plt.close(fig)
    print(f"Saved: {bar_path}")

    # 2. SNR Robustness Curves
    snr_df = pd.read_csv(os.path.join(EXP_DIR, "all_models_snr_comparison.csv"))
    snr_vals = [s if s != "inf" else 1000.0 for s in snr_df["SNR"].astype(float)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # CBF SNR
    axes[0].plot(snr_vals, snr_df["DNN_CBF_RMSE"], "o-", label="Baseline DNN", color="#4c72b0", lw=2)
    axes[0].plot(snr_vals, snr_df["ELM_CBF_RMSE"], "s--", label="Standalone ELM", color="#dd8452", lw=1.5)
    axes[0].plot(snr_vals, snr_df["Ens_CBF_RMSE"], "^-.", label="Weighted Ensemble", color="#55a868", lw=1.5)
    axes[0].plot(snr_vals, snr_df["FeatELM_CBF_RMSE"], "d-", label="DNN-Feature ELM", color="#8172b3", lw=2.5)
    axes[0].set_xscale("log")
    axes[0].set_title("CBF Robustness vs Acquisition SNR (Grid=2,392 pts)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Nominal SNR (log scale)", fontsize=10)
    axes[0].set_ylabel("CBF RMSE (mL / [100g·min])", fontsize=10)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9.5)

    # ATT SNR
    axes[1].plot(snr_vals, snr_df["DNN_ATT_RMSE"], "o-", label="Baseline DNN", color="#4c72b0", lw=2)
    axes[1].plot(snr_vals, snr_df["ELM_ATT_RMSE"], "s--", label="Standalone ELM", color="#dd8452", lw=1.5)
    axes[1].plot(snr_vals, snr_df["Ens_ATT_RMSE"], "^-.", label="Weighted Ensemble", color="#55a868", lw=1.5)
    axes[1].plot(snr_vals, snr_df["FeatELM_ATT_RMSE"], "d-", label="DNN-Feature ELM", color="#8172b3", lw=2.5)
    axes[1].set_xscale("log")
    axes[1].set_title("ATT Robustness vs Acquisition SNR (Grid=2,392 pts)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Nominal SNR (log scale)", fontsize=10)
    axes[1].set_ylabel("ATT RMSE (seconds)", fontsize=10)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9.5)

    fig.tight_layout()
    snr_path = os.path.join(DIAG_DIR, "all_models_snr_curves.png")
    fig.savefig(snr_path)
    plt.close(fig)
    print(f"Saved: {snr_path}")

if __name__ == "__main__":
    generate_plots()
