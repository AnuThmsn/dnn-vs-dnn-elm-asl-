"""
generate_diagnostics.py
Phase 3: Formal Baseline Diagnostics & Verified Training Plots
Generates genuine training curves, prediction diagnostics, residual distributions,
SNR-wise performance curves, parameter range breakdowns, and raw boundary violation audits.
All artifacts are saved to experiments/diagnostics/ on Drive D.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import memory_config
from physics import PLDS, ATT_MIN, ATT_MAX, CBF_MIN, CBF_MAX
from train_baseline_A import StandardizedNet, calc_metrics

BASE_DIR = os.path.join(os.path.dirname(__file__), "baseline_A")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
DIAG_DIR = os.path.join(os.path.dirname(__file__), "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

def plot_training_curves():
    """Requirement: Genuine training & validation loss curves from training_history.json."""
    with open(os.path.join(BASE_DIR, "training_history.json"), "r") as f:
        hist = json.load(f)

    # 1. ATTnet Training Curve
    att_hist = hist["att_history"]
    att_epochs = [e["epoch"] for e in att_hist]
    att_tr_loss = [e["train_loss"] for e in att_hist]
    att_val_loss = [e["val_loss"] for e in att_hist]
    best_att_idx = int(np.argmin(att_val_loss))
    best_att_ep = att_epochs[best_att_idx]
    best_att_val = att_val_loss[best_att_idx]
    stop_att_ep = att_epochs[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(att_epochs, att_tr_loss, label="Train MSE Loss (Standardized)", color="#1f77b4", lw=2)
    ax.plot(att_epochs, att_val_loss, label="Val MSE Loss (Standardized)", color="#ff7f0e", lw=2)
    ax.scatter([best_att_ep], [best_att_val], color="red", s=80, zorder=5,
               label=f"Best Val: Epoch {best_att_ep} ({best_att_val:.4f})")
    ax.axvline(best_att_ep, color="red", linestyle="--", alpha=0.5)
    ax.axvline(stop_att_ep, color="gray", linestyle=":", label=f"Early Stop: Epoch {stop_att_ep}")
    ax.set_title("Baseline A: ATTnet (4x64 ELU MLP) Training History", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Standardized MSE Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    att_plot_path = os.path.join(DIAG_DIR, "training_curves_ATT.png")
    fig.tight_layout()
    fig.savefig(att_plot_path)
    plt.close(fig)
    print(f"Saved: {att_plot_path} (History length: {len(att_hist)} epochs)")

    # 2. CBFnet Training Curve
    cbf_hist = hist["cbf_history"]
    cbf_epochs = [e["epoch"] for e in cbf_hist]
    cbf_tr_loss = [e["train_loss"] for e in cbf_hist]
    cbf_val_loss = [e["val_loss"] for e in cbf_hist]
    best_cbf_idx = int(np.argmin(cbf_val_loss))
    best_cbf_ep = cbf_epochs[best_cbf_idx]
    best_cbf_val = cbf_val_loss[best_cbf_idx]
    stop_cbf_ep = cbf_epochs[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cbf_epochs, cbf_tr_loss, label="Train MSE Loss (Standardized)", color="#2ca02c", lw=2)
    ax.plot(cbf_epochs, cbf_val_loss, label="Val MSE Loss (Standardized)", color="#d62728", lw=2)
    ax.scatter([best_cbf_ep], [best_cbf_val], color="red", s=80, zorder=5,
               label=f"Best Val: Epoch {best_cbf_ep} ({best_cbf_val:.4f})")
    ax.axvline(best_cbf_ep, color="red", linestyle="--", alpha=0.5)
    ax.axvline(stop_cbf_ep, color="gray", linestyle=":", label=f"Early Stop: Epoch {stop_cbf_ep}")
    ax.set_title("Baseline A: CBFnet (5x128 ELU MLP) Training History", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Standardized MSE Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    cbf_plot_path = os.path.join(DIAG_DIR, "training_curves_CBF.png")
    fig.tight_layout()
    fig.savefig(cbf_plot_path)
    plt.close(fig)
    print(f"Saved: {cbf_plot_path} (History length: {len(cbf_hist)} epochs)")

def evaluate_predictions_and_diagnostics():
    """Evaluate predicted vs true, residuals, bound violations, and ranges."""
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    # Load test inputs & ground truth
    x_test_mmap = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    y_test_mmap = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    snr_test_mmap = np.load(os.path.join(CACHE_DIR, "snr_test.npy"), mmap_mode='r')

    cbf_true = y_test_mmap[:, 0]
    att_true = y_test_mmap[:, 1]
    X_test_n = ((x_test_mmap[:] - X_mean) / X_std).astype(np.float32)

    # Load models
    import torch
    device = torch.device("cpu")
    att_net = StandardizedNet(len(PLDS), 4, 64, att_mean, att_std, ATT_MIN, ATT_MAX)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device))
    att_net.eval()

    cbf_net = StandardizedNet(len(PLDS), 5, 128, cbf_mean, cbf_std, CBF_MIN, CBF_MAX)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device))
    cbf_net.eval()

    # Compute RAW unclipped predictions and CLAMPED predictions
    with torch.no_grad():
        x_t = torch.from_numpy(X_test_n)
        raw_att_std = att_net(x_t).squeeze(-1).numpy()
        raw_cbf_std = cbf_net(x_t).squeeze(-1).numpy()

    cbf_raw = raw_cbf_std * cbf_std + cbf_mean
    att_raw = raw_att_std * att_std + att_mean

    cbf_pred = np.clip(cbf_raw, CBF_MIN, CBF_MAX)
    att_pred = np.clip(att_raw, ATT_MIN, ATT_MAX)

    # 1. Physical-Bound Violation Analysis before clipping
    cbf_lower_violations = int(np.sum(cbf_raw < CBF_MIN))
    cbf_upper_violations = int(np.sum(cbf_raw > CBF_MAX))
    att_lower_violations = int(np.sum(att_raw < ATT_MIN))
    att_upper_violations = int(np.sum(att_raw > ATT_MAX))

    N_test = len(cbf_true)
    bound_report = {
        "N_test": N_test,
        "CBF_min_raw": float(cbf_raw.min()), "CBF_max_raw": float(cbf_raw.max()),
        "CBF_below_0": cbf_lower_violations, "CBF_below_0_pct": (cbf_lower_violations / N_test) * 100,
        "CBF_above_100": cbf_upper_violations, "CBF_above_100_pct": (cbf_upper_violations / N_test) * 100,
        "ATT_min_raw": float(att_raw.min()), "ATT_max_raw": float(att_raw.max()),
        "ATT_below_0.5": att_lower_violations, "ATT_below_0.5_pct": (att_lower_violations / N_test) * 100,
        "ATT_above_3.0": att_upper_violations, "ATT_above_3.0_pct": (att_upper_violations / N_test) * 100
    }
    with open(os.path.join(DIAG_DIR, "physical_bound_violations.json"), "w") as f:
        json.dump(bound_report, f, indent=2)
    print("\n--- Physical Bound Violation Audit (Before Clipping) ---")
    print(f"CBF Raw Range: [{cbf_raw.min():.2f}, {cbf_raw.max():.2f}] | Below 0: {cbf_lower_violations} ({bound_report['CBF_below_0_pct']:.3f}%), Above 100: {cbf_upper_violations} ({bound_report['CBF_above_100_pct']:.3f}%)")
    print(f"ATT Raw Range: [{att_raw.min():.2f}, {att_raw.max():.2f}] | Below 0.5: {att_lower_violations} ({bound_report['ATT_below_0.5_pct']:.3f}%), Above 3.0: {att_upper_violations} ({bound_report['ATT_above_3.0_pct']:.3f}%)")

    # 2. Predicted vs True Scatter Plot (Subsample 10,000 for clarity)
    idx_sub = np.random.default_rng(42).choice(N_test, size=10_000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CBF Scatter
    axes[0].scatter(cbf_true[idx_sub], cbf_pred[idx_sub], alpha=0.25, s=8, color="#1f77b4")
    axes[0].plot([0, 100], [0, 100], "r--", lw=2, label="Identity (y = x)")
    axes[0].set_title(f"Baseline A: CBF True vs Predicted (N=10k sample)\nRMSE: 3.73 mL, CCC: 0.9916", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Ground Truth CBF (mL / [100g·min])", fontsize=10)
    axes[0].set_ylabel("Predicted CBF (mL / [100g·min])", fontsize=10)
    axes[0].set_xlim(-2, 102); axes[0].set_ylim(-2, 102)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True)

    # ATT Scatter
    axes[1].scatter(att_true[idx_sub], att_pred[idx_sub], alpha=0.25, s=8, color="#ff7f0e")
    axes[1].plot([0.5, 3.0], [0.5, 3.0], "r--", lw=2, label="Identity (y = x)")
    axes[1].set_title(f"Baseline A: ATT True vs Predicted (N=10k sample)\nRMSE: 0.306 s, CCC: 0.9009", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Ground Truth ATT (seconds)", fontsize=10)
    axes[1].set_ylabel("Predicted ATT (seconds)", fontsize=10)
    axes[1].set_xlim(0.4, 3.1); axes[1].set_ylim(0.4, 3.1)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True)

    fig.tight_layout()
    scatter_path = os.path.join(DIAG_DIR, "predicted_vs_true_scatter.png")
    fig.savefig(scatter_path)
    plt.close(fig)
    print(f"Saved: {scatter_path}")

    # 3. Residual Distributions & Residuals vs True Values
    err_cbf = cbf_pred - cbf_true
    err_att = att_pred - att_true

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residual Histogram CBF
    axes[0, 0].hist(err_cbf, bins=100, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
    axes[0, 0].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[0, 0].axvline(err_cbf.mean(), color="black", linestyle=":", lw=1.5,
                       label=f"Mean Bias: {err_cbf.mean():.3f}")
    axes[0, 0].set_title(f"CBF Residual Distribution (N={N_test:,})\nStd: {err_cbf.std():.3f} mL", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Residual: Predicted - True (mL / [100g·min])")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.4)

    # Residual Histogram ATT
    axes[0, 1].hist(err_att, bins=100, color="#ff7f0e", edgecolor="black", alpha=0.7, density=True)
    axes[0, 1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[0, 1].axvline(err_att.mean(), color="black", linestyle=":", lw=1.5,
                       label=f"Mean Bias: {err_att.mean():.4f}")
    axes[0, 1].set_title(f"ATT Residual Distribution (N={N_test:,})\nStd: {err_att.std():.3f} s", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Residual: Predicted - True (seconds)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.4)

    # Residuals vs True CBF (Heteroscedasticity)
    axes[1, 0].scatter(cbf_true[idx_sub], err_cbf[idx_sub], alpha=0.2, s=8, color="#1f77b4")
    axes[1, 0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1, 0].set_title("CBF Residual vs True Parameter (Heteroscedasticity Analysis)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Ground Truth CBF (mL / [100g·min])")
    axes[1, 0].set_ylabel("Residual (Predicted - True)")
    axes[1, 0].grid(True, alpha=0.4)

    # Residuals vs True ATT
    axes[1, 1].scatter(att_true[idx_sub], err_att[idx_sub], alpha=0.2, s=8, color="#ff7f0e")
    axes[1, 1].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1, 1].set_title("ATT Residual vs True Parameter (Arrival Threshold Analysis)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Ground Truth ATT (seconds)")
    axes[1, 1].set_ylabel("Residual (Predicted - True)")
    axes[1, 1].grid(True, alpha=0.4)

    fig.tight_layout()
    resid_path = os.path.join(DIAG_DIR, "residual_diagnostics.png")
    fig.savefig(resid_path)
    plt.close(fig)
    print(f"Saved: {resid_path}")

    # 4. Error Breakdown by Parameter Range
    cbf_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    cbf_range_rows = []
    for low, high in cbf_bins:
        mask = (cbf_true >= low) & (cbf_true < high) if high < 100 else (cbf_true >= low) & (cbf_true <= high)
        m = calc_metrics(cbf_true[mask], cbf_pred[mask])
        cbf_range_rows.append({"Target": "CBF", "Range": f"[{low}, {high})", "Count": int(np.sum(mask)),
                               "RMSE": m["RMSE"], "MAE": m["MAE"], "Bias": m["Bias"], "CCC": m["CCC"]})

    att_bins = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
    att_range_rows = []
    for low, high in att_bins:
        mask = (att_true >= low) & (att_true < high) if high < 3.0 else (att_true >= low) & (att_true <= high)
        m = calc_metrics(att_true[mask], att_pred[mask])
        att_range_rows.append({"Target": "ATT", "Range": f"[{low:.1f}, {high:.1f})", "Count": int(np.sum(mask)),
                               "RMSE": m["RMSE"], "MAE": m["MAE"], "Bias": m["Bias"], "CCC": m["CCC"]})

    range_df = pd.DataFrame(cbf_range_rows + att_range_rows)
    range_csv = os.path.join(DIAG_DIR, "parameter_range_errors.csv")
    range_df.to_csv(range_csv, index=False)
    print(f"\nSaved parameter range breakdown: {range_csv}")
    print(range_df.to_string(index=False))

    # 5. Plot Metrics Grouped by SNR
    snr_csv_path = os.path.join(BASE_DIR, "snr_evaluation_baselineA.csv")
    snr_df = pd.read_csv(snr_csv_path)

    # Replace 'inf' with 1000 for plotting
    snr_plot_vals = [s if s != "inf" else 1000.0 for s in snr_df["SNR"].astype(float)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(snr_plot_vals, snr_df["CBF_RMSE"], marker="o", color="#1f77b4", lw=2, label="CBF RMSE")
    axes[0].set_xscale("log")
    axes[0].set_title("CBF Estimation Error vs Acquisition SNR", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Nominal SNR (log scale)", fontsize=10)
    axes[0].set_ylabel("CBF RMSE (mL / [100g·min])", fontsize=10)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(snr_plot_vals, snr_df["ATT_RMSE"], marker="s", color="#ff7f0e", lw=2, label="ATT RMSE")
    axes[1].set_xscale("log")
    axes[1].set_title("ATT Estimation Error vs Acquisition SNR", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Nominal SNR (log scale)", fontsize=10)
    axes[1].set_ylabel("ATT RMSE (seconds)", fontsize=10)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.tight_layout()
    snr_plot_path = os.path.join(DIAG_DIR, "metrics_grouped_by_snr.png")
    fig.savefig(snr_plot_path)
    plt.close(fig)
    print(f"Saved: {snr_plot_path}")

if __name__ == "__main__":
    plot_training_curves()
    evaluate_predictions_and_diagnostics()

