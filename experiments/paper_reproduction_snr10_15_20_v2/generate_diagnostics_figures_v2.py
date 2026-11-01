"""
generate_diagnostics_figures_v2.py
Publication-quality diagnostic figures for V2 study:
Generates all 8 required diagnostic figures + 3-way model comparison figures (PNG 300 DPI + vector PDF):
1. DNN residual distribution: training vs calibration data
2. H-DELM correction distribution: calibration vs test data
3. Residual error before and after ELM correction (boxplots & KDEs)
4. CBF ground-truth vs prediction scatter (DNN-G, V1, V2)
5. ATT ground-truth vs prediction scatter (DNN-G, V1, V2)
6. Error vs ground-truth parameter (binned bias & variance)
7. Error vs SNR plots (RMSE, MAE, CCC across SNR 10, 15, 20)
8. Ridge lambda hyperparameter sensitivity curve
"""

import os
import numpy as np
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
    "figure.autolayout": True
})

def save_fig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {name} (PNG & PDF)")

def plot_all_diagnostics(
    diag_data,
    metrics_comp_df,
    lambda_cbf_df,
    lambda_att_df,
    output_dir
):
    print("Generating comprehensive diagnostic figures for V2...")

    # -------------------------------------------------------------
    # 1. DNN Residual Distribution on Train vs Calibration Data
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # CBF residuals
    ax = axes[0]
    ax.hist(diag_data["r_cbf_train"], bins=100, density=True, alpha=0.5, color="#1f77b4", label="DNN Training Data (1M)")
    ax.hist(diag_data["r_cbf_cal"], bins=100, density=True, alpha=0.5, color="#ff7f0e", label="Independent Calibration (100k)")
    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("CBF Residual Distribution: $r = y_{\\mathrm{true}} - y_{\\mathrm{DNN}}$")
    ax.set_xlabel("Residual (mL/100g/min)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim([-25, 25])
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # ATT residuals
    ax = axes[1]
    ax.hist(diag_data["r_att_train"], bins=100, density=True, alpha=0.5, color="#1f77b4", label="DNN Training Data (1M)")
    ax.hist(diag_data["r_att_cal"], bins=100, density=True, alpha=0.5, color="#ff7f0e", label="Independent Calibration (100k)")
    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("ATT Residual Distribution: $r = y_{\\mathrm{true}} - y_{\\mathrm{DNN}}$")
    ax.set_xlabel("Residual (s)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim([-1.5, 1.5])
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    save_fig(fig, output_dir, "01_dnn_residual_distribution_train_vs_cal")

    # -------------------------------------------------------------
    # 2. H-DELM Correction Distribution on Calibration vs Test Data
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # CBF corrections
    ax = axes[0]
    ax.hist(diag_data["corr_cbf_cal"], bins=100, density=True, alpha=0.5, color="#2ca02c", label="Calibration Set (100k)")
    ax.hist(diag_data["corr_cbf_test"], bins=100, density=True, alpha=0.5, color="#9467bd", label="Test Set (30k)")
    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("H-DELM V2 CBF Correction Distribution: $\\hat{r}$")
    ax.set_xlabel("Predicted Correction $\\hat{r}$ (mL/100g/min)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim([-15, 15])
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # ATT corrections
    ax = axes[1]
    ax.hist(diag_data["corr_att_cal"], bins=100, density=True, alpha=0.5, color="#2ca02c", label="Calibration Set (100k)")
    ax.hist(diag_data["corr_att_test"], bins=100, density=True, alpha=0.5, color="#9467bd", label="Test Set (30k)")
    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("H-DELM V2 ATT Correction Distribution: $\\hat{r}$")
    ax.set_xlabel("Predicted Correction $\\hat{r}$ (s)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim([-0.5, 0.5])
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    save_fig(fig, output_dir, "02_hdelm_correction_distribution_cal_vs_test")

    # -------------------------------------------------------------
    # 3. Residual Error Before and After ELM Correction on Test Set
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # CBF Error before vs after
    ax = axes[0]
    e_cbf_before = diag_data["cbf_dnn_test"] - diag_data["cbf_true_test"]
    e_cbf_after = diag_data["cbf_v2_test"] - diag_data["cbf_true_test"]
    ax.hist(e_cbf_before, bins=100, density=True, alpha=0.5, color="#d62728", label=f"DNN-G (RMSE={np.sqrt(np.mean(e_cbf_before**2)):.2f})")
    ax.hist(e_cbf_after, bins=100, density=True, alpha=0.5, color="#1f77b4", label=f"H-DELM V2 (RMSE={np.sqrt(np.mean(e_cbf_after**2)):.2f})")
    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("CBF Error Distribution Before vs After H-DELM V2")
    ax.set_xlabel("Error: $\\hat{y} - y_{\\mathrm{true}}$ (mL/100g/min)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim([-15, 15])
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # ATT Error before vs after
    ax = axes[1]
    e_att_before = diag_data["att_dnn_test"] - diag_data["att_true_test"]
    e_att_after = diag_data["att_v2_test"] - diag_data["att_true_test"]
    ax.hist(e_att_before, bins=100, density=True, alpha=0.5, color="#d62728", label=f"DNN-G (RMSE={np.sqrt(np.mean(e_att_before**2)):.3f})")
    ax.hist(e_att_after, bins=100, density=True, alpha=0.5, color="#1f77b4", label=f"H-DELM V2 (RMSE={np.sqrt(np.mean(e_att_after**2)):.3f})")
    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title("ATT Error Distribution Before vs After H-DELM V2")
    ax.set_xlabel("Error: $\\hat{y} - y_{\\mathrm{true}}$ (s)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim([-1.2, 1.2])
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    save_fig(fig, output_dir, "03_residual_error_before_after_correction")

    # -------------------------------------------------------------
    # 4. CBF Ground-Truth vs Predictions Scatter (DNN-G, V1, V2)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    models = [("DNN-G Baseline", diag_data["cbf_dnn_test"], "#1f77b4"),
              ("H-DELM V1 (DNN 1M Fit)", diag_data["cbf_v1_test"], "#ff7f0e"),
              ("H-DELM V2 (Indep 100k Fit)", diag_data["cbf_v2_test"], "#2ca02c")]

    y_t = diag_data["cbf_true_test"]
    sub_idx = np.random.default_rng(42).choice(len(y_t), size=3000, replace=False)

    for ax, (m_name, m_pred, col) in zip(axes, models):
        ax.scatter(y_t[sub_idx], m_pred[sub_idx], s=6, alpha=0.25, color=col)
        ax.plot([0, 100], [0, 100], "k--", lw=1.5, label="Identity Line")
        r_val = float(np.corrcoef(y_t, m_pred)[0, 1])
        rmse_val = float(np.sqrt(np.mean((m_pred - y_t)**2)))
        ax.set_title(f"{m_name}\nRMSE={rmse_val:.3f}, r={r_val:.4f}")
        ax.set_xlabel("True CBF (mL/100g/min)")
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Estimated CBF (mL/100g/min)")
    save_fig(fig, output_dir, "04_cbf_prediction_scatter_3way")

    # -------------------------------------------------------------
    # 5. ATT Ground-Truth vs Predictions Scatter (DNN-G, V1, V2)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    models_att = [("DNN-G Baseline", diag_data["att_dnn_test"], "#1f77b4"),
                  ("H-DELM V1 (DNN 1M Fit)", diag_data["att_v1_test"], "#ff7f0e"),
                  ("H-DELM V2 (Indep 100k Fit)", diag_data["att_v2_test"], "#2ca02c")]

    y_t_att = diag_data["att_true_test"]
    for ax, (m_name, m_pred, col) in zip(axes, models_att):
        ax.scatter(y_t_att[sub_idx], m_pred[sub_idx], s=6, alpha=0.25, color=col)
        ax.plot([0.5, 3.0], [0.5, 3.0], "k--", lw=1.5, label="Identity Line")
        r_val = float(np.corrcoef(y_t_att, m_pred)[0, 1])
        rmse_val = float(np.sqrt(np.mean((m_pred - y_t_att)**2)))
        ax.set_title(f"{m_name}\nRMSE={rmse_val:.3f} s, r={r_val:.4f}")
        ax.set_xlabel("True ATT (s)")
        ax.set_xlim([0.5, 3.0])
        ax.set_ylim([0.5, 3.0])
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Estimated ATT (s)")
    save_fig(fig, output_dir, "05_att_prediction_scatter_3way")

    # -------------------------------------------------------------
    # 6. Error vs Ground-Truth Parameter Plots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # CBF error vs True CBF
    ax = axes[0]
    bins_cbf = np.linspace(0, 100, 11)
    centers_cbf = 0.5 * (bins_cbf[:-1] + bins_cbf[1:])
    cbf_bin_idx = np.digitize(y_t, bins_cbf) - 1
    
    dnn_cbf_mae_bins = [np.mean(np.abs(diag_data["cbf_dnn_test"][cbf_bin_idx == b] - y_t[cbf_bin_idx == b])) for b in range(10)]
    v1_cbf_mae_bins = [np.mean(np.abs(diag_data["cbf_v1_test"][cbf_bin_idx == b] - y_t[cbf_bin_idx == b])) for b in range(10)]
    v2_cbf_mae_bins = [np.mean(np.abs(diag_data["cbf_v2_test"][cbf_bin_idx == b] - y_t[cbf_bin_idx == b])) for b in range(10)]

    ax.plot(centers_cbf, dnn_cbf_mae_bins, "o-", color="#1f77b4", label="DNN-G")
    ax.plot(centers_cbf, v1_cbf_mae_bins, "s--", color="#ff7f0e", label="H-DELM V1")
    ax.plot(centers_cbf, v2_cbf_mae_bins, "^-", color="#2ca02c", label="H-DELM V2")
    ax.set_title("CBF Estimation Error vs True CBF")
    ax.set_xlabel("True CBF (mL/100g/min)")
    ax.set_ylabel("Mean Absolute Error (mL/100g/min)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # ATT error vs True ATT
    ax = axes[1]
    bins_att = np.linspace(0.5, 3.0, 11)
    centers_att = 0.5 * (bins_att[:-1] + bins_att[1:])
    att_bin_idx = np.digitize(y_t_att, bins_att) - 1

    dnn_att_mae_bins = [np.mean(np.abs(diag_data["att_dnn_test"][att_bin_idx == b] - y_t_att[att_bin_idx == b])) for b in range(10)]
    v1_att_mae_bins = [np.mean(np.abs(diag_data["att_v1_test"][att_bin_idx == b] - y_t_att[att_bin_idx == b])) for b in range(10)]
    v2_att_mae_bins = [np.mean(np.abs(diag_data["att_v2_test"][att_bin_idx == b] - y_t_att[att_bin_idx == b])) for b in range(10)]

    ax.plot(centers_att, dnn_att_mae_bins, "o-", color="#1f77b4", label="DNN-G")
    ax.plot(centers_att, v1_att_mae_bins, "s--", color="#ff7f0e", label="H-DELM V1")
    ax.plot(centers_att, v2_att_mae_bins, "^-", color="#2ca02c", label="H-DELM V2")
    ax.set_title("ATT Estimation Error vs True ATT")
    ax.set_xlabel("True ATT (s)")
    ax.set_ylabel("Mean Absolute Error (s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    save_fig(fig, output_dir, "06_error_vs_ground_truth_parameter")

    # -------------------------------------------------------------
    # 7. Error vs SNR Plots (RMSE across SNR 10, 15, 20)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    snrs = [10, 15, 20]
    
    # CBF RMSE
    ax = axes[0]
    cbf_df = metrics_comp_df[(metrics_comp_df["Parameter"] == "CBF") & (metrics_comp_df["SNR"].isin(snrs))]
    dnn_cbf_rmse = [cbf_df[(cbf_df["Model"] == "DNN-G") & (cbf_df["SNR"] == s)]["RMSE"].values[0] for s in snrs]
    v1_cbf_rmse = [cbf_df[(cbf_df["Model"] == "H-DELM V1") & (cbf_df["SNR"] == s)]["RMSE"].values[0] for s in snrs]
    v2_cbf_rmse = [cbf_df[(cbf_df["Model"] == "H-DELM V2") & (cbf_df["SNR"] == s)]["RMSE"].values[0] for s in snrs]

    ax.plot(snrs, dnn_cbf_rmse, "o-", color="#1f77b4", label="DNN-G Baseline")
    ax.plot(snrs, v1_cbf_rmse, "s--", color="#ff7f0e", label="H-DELM V1 (1M Train Fit)")
    ax.plot(snrs, v2_cbf_rmse, "^-", color="#2ca02c", label="H-DELM V2 (100k Cal Fit)")
    ax.set_title("CBF Estimation Error vs SNR")
    ax.set_xlabel("Target SNR (dB)")
    ax.set_ylabel("RMSE (mL/100g/min)")
    ax.set_xticks(snrs)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # ATT RMSE
    ax = axes[1]
    att_df = metrics_comp_df[(metrics_comp_df["Parameter"] == "ATT") & (metrics_comp_df["SNR"].isin(snrs))]
    dnn_att_rmse = [att_df[(att_df["Model"] == "DNN-G") & (att_df["SNR"] == s)]["RMSE"].values[0] for s in snrs]
    v1_att_rmse = [att_df[(att_df["Model"] == "H-DELM V1") & (att_df["SNR"] == s)]["RMSE"].values[0] for s in snrs]
    v2_att_rmse = [att_df[(att_df["Model"] == "H-DELM V2") & (att_df["SNR"] == s)]["RMSE"].values[0] for s in snrs]

    ax.plot(snrs, dnn_att_rmse, "o-", color="#1f77b4", label="DNN-G Baseline")
    ax.plot(snrs, v1_att_rmse, "s--", color="#ff7f0e", label="H-DELM V1 (1M Train Fit)")
    ax.plot(snrs, v2_att_rmse, "^-", color="#2ca02c", label="H-DELM V2 (100k Cal Fit)")
    ax.set_title("ATT Estimation Error vs SNR")
    ax.set_xlabel("Target SNR (dB)")
    ax.set_ylabel("RMSE (s)")
    ax.set_xticks(snrs)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    save_fig(fig, output_dir, "07_error_vs_snr")

    # -------------------------------------------------------------
    # 8. Ridge Lambda Hyperparameter Sensitivity Curve
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # CBF Lambda sensitivity
    ax = axes[0]
    ax.semilogx(lambda_cbf_df["ridge_lambda"], lambda_cbf_df["val_rmse"], "o-", color="#1f77b4", label="Validation RMSE")
    ax.set_title("CBF ELM Ridge $\\lambda$ Sensitivity")
    ax.set_xlabel("Ridge Regularization Parameter $\\lambda$")
    ax.set_ylabel("Validation RMSE (mL/100g/min)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # ATT Lambda sensitivity
    ax = axes[1]
    ax.semilogx(lambda_att_df["ridge_lambda"], lambda_att_df["val_rmse"], "s-", color="#ff7f0e", label="Validation RMSE")
    ax.set_title("ATT ELM Ridge $\\lambda$ Sensitivity")
    ax.set_xlabel("Ridge Regularization Parameter $\\lambda$")
    ax.set_ylabel("Validation RMSE (s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    save_fig(fig, output_dir, "08_lambda_sensitivity_curve")

    # -------------------------------------------------------------
    # 9. V1 vs V2 Relative Improvement Comparison Bar Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # CBF
    ax = axes[0]
    x_pos = np.arange(len(snrs))
    w = 0.35
    
    v1_gain_cbf = [((d - v) / d) * 100.0 for d, v in zip(dnn_cbf_rmse, v1_cbf_rmse)]
    v2_gain_cbf = [((d - v) / d) * 100.0 for d, v in zip(dnn_cbf_rmse, v2_cbf_rmse)]

    ax.bar(x_pos - w/2, v1_gain_cbf, width=w, color="#ff7f0e", label="H-DELM V1 (1M Train Fit)")
    ax.bar(x_pos + w/2, v2_gain_cbf, width=w, color="#2ca02c", label="H-DELM V2 (100k Cal Fit)")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"SNR {s} dB" for s in snrs])
    ax.set_title("CBF RMSE Improvement Relative to DNN-G (%)")
    ax.set_ylabel("Relative Improvement (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # ATT
    ax = axes[1]
    v1_gain_att = [((d - v) / d) * 100.0 for d, v in zip(dnn_att_rmse, v1_att_rmse)]
    v2_gain_att = [((d - v) / d) * 100.0 for d, v in zip(dnn_att_rmse, v2_att_rmse)]

    ax.bar(x_pos - w/2, v1_gain_att, width=w, color="#ff7f0e", label="H-DELM V1 (1M Train Fit)")
    ax.bar(x_pos + w/2, v2_gain_att, width=w, color="#2ca02c", label="H-DELM V2 (100k Cal Fit)")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"SNR {s} dB" for s in snrs])
    ax.set_title("ATT RMSE Improvement Relative to DNN-G (%)")
    ax.set_ylabel("Relative Improvement (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    save_fig(fig, output_dir, "09_v1_vs_v2_percentage_gain_comparison")

    print("All diagnostic figures generated successfully.")

