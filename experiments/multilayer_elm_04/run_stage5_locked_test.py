"""
run_stage5_locked_test.py
Stage 5: Final Locked-Test Cohort Evaluation for Experiment 4.
Evaluates the frozen candidate models on the locked test cohort (N=100,000)
using strictly original PLDs [1.525, 2.025, 2.525, 3.025] s.
Computes:
- Comprehensive metrics: RMSE, MAE, Bias, CCC, R2
- Paired statistical tests and 95% bootstrap confidence intervals
- SNR-stratified metrics across all 12 SNR levels
- ATT physiological interval breakdown
- Publication-quality figures
Outputs:
- final_test_metrics.csv
- final_test_snr_stratified.csv
- final_test_att_intervals.csv
- plots/final_test_evaluation.png
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
from scipy import stats

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
print(f"[Stage 5 Locked Test] Device: {device}")

def bootstrap_paired_diff(y_true, pred_base, pred_cand, n_boot=2000, seed=42):
    """Compute 95% bootstrap CI for paired difference in RMSE (cand - base)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rmse_base = np.sqrt(np.mean((pred_base[idx] - y_true[idx]) ** 2))
        rmse_cand = np.sqrt(np.mean((pred_cand[idx] - y_true[idx]) ** 2))
        diffs.append(rmse_cand - rmse_base)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    return ci_lower, ci_upper

def paired_statistical_tests(y_true, pred_base, pred_cand, seed=42):
    """Paired t-test on squared errors and Wilcoxon on absolute errors (10,000 subsample)."""
    err_base_sq = (pred_base - y_true) ** 2
    err_cand_sq = (pred_cand - y_true) ** 2
    
    # Paired t-test on full cohort for MSE difference
    t_stat, p_val_t = stats.ttest_rel(err_cand_sq, err_base_sq)
    
    # Wilcoxon signed-rank on 10,000 observations (subsampled strictly with seed 42)
    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(len(y_true), size=min(10_000, len(y_true)), replace=False)
    abs_base = np.abs(pred_base[sub_idx] - y_true[sub_idx])
    abs_cand = np.abs(pred_cand[sub_idx] - y_true[sub_idx])
    w_stat, p_val_w = stats.wilcoxon(abs_cand, abs_base)

    return float(t_stat), float(p_val_t), float(w_stat), float(p_val_w)

def main():
    print("=" * 80)
    print("STAGE 5: FINAL LOCKED-TEST EVALUATION (N=100,000)")
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

    # Load Training Data for Fitting Models
    X_tr_raw = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    Y_tr_raw = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    X_tr_std = (X_tr_raw[:] - X_mean) / X_std
    Y_cbf_tr_std = (Y_tr_raw[:, 0] - y_cbf_mean) / y_cbf_std
    Y_att_tr_std = (Y_tr_raw[:, 1] - y_att_mean) / y_att_std

    # Load Locked Test Data
    X_te_raw = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    Y_te_raw = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    SNR_te = np.load(os.path.join(CACHE_DIR, "SNR_test.npy"), mmap_mode='r')

    X_te_std = (X_te_raw[:] - X_mean) / X_std
    Y_cbf_te = Y_te_raw[:, 0].astype(np.float64)
    Y_att_te = Y_te_raw[:, 1].astype(np.float64)

    # 2. Train and Predict Candidates on Locked Test Set
    print("\n[1/5] Evaluating Baseline A (DNN-G)...")
    pred_cbf_dnn = cbf_net.predict_physical(X_te_std)
    pred_att_dnn = att_net.predict_physical(X_te_std)

    print("[2/5] Training and Evaluating Historical Feature ELM (Rep A, 512, alpha=1e-2)...")
    m_hist_cbf = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_hist_cbf.accumulate_HtH_HtY(cbf_extractor, "Rep_A", X_tr_std, Y_cbf_tr_std)
    beta = m_hist_cbf.solve_alpha(HtH, HtY, 1e-2)
    pred_cbf_hist = np.clip(m_hist_cbf.predict_from_beta(cbf_extractor, "Rep_A", X_te_std, beta) * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
    del HtH, HtY, beta

    m_hist_att = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_hist_att.accumulate_HtH_HtY(att_extractor, "Rep_A", X_tr_std, Y_att_tr_std)
    beta = m_hist_att.solve_alpha(HtH, HtY, 1e-2)
    pred_att_hist = np.clip(m_hist_att.predict_from_beta(att_extractor, "Rep_A", X_te_std, beta) * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
    del HtH, HtY, beta

    print("[3/5] Training and Evaluating Penultimate Tuned ELM (Rep A, 512, alpha=1.0 for CBF; alpha=10.0 for ATT)...")
    m_penult_cbf = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_penult_cbf.accumulate_HtH_HtY(cbf_extractor, "Rep_A", X_tr_std, Y_cbf_tr_std)
    beta = m_penult_cbf.solve_alpha(HtH, HtY, 1.0)
    pred_cbf_penult = np.clip(m_penult_cbf.predict_from_beta(cbf_extractor, "Rep_A", X_te_std, beta) * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
    del HtH, HtY, beta

    m_penult_att = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_penult_att.accumulate_HtH_HtY(att_extractor, "Rep_A", X_tr_std, Y_att_tr_std)
    beta = m_penult_att.solve_alpha(HtH, HtY, 10.0)
    pred_att_penult = np.clip(m_penult_att.predict_from_beta(att_extractor, "Rep_A", X_te_std, beta) * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
    del HtH, HtY, beta

    print("[4/5] Training and Evaluating Multi-Layer Feature ELM (Rep C: [h1, h5] for CBF; [h1, h4] for ATT)...")
    m_multi_cbf = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_multi_cbf.accumulate_HtH_HtY(cbf_extractor, "Rep_C", X_tr_std, Y_cbf_tr_std)
    beta = m_multi_cbf.solve_alpha(HtH, HtY, 0.1)
    pred_cbf_multi = np.clip(m_multi_cbf.predict_from_beta(cbf_extractor, "Rep_C", X_te_std, beta) * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
    del HtH, HtY, beta

    m_multi_att = ReadoutModel("elm", "tanh", 512, seed=42)
    HtH, HtY = m_multi_att.accumulate_HtH_HtY(att_extractor, "Rep_C", X_tr_std, Y_att_tr_std)
    beta = m_multi_att.solve_alpha(HtH, HtY, 10.0)
    pred_att_multi = np.clip(m_multi_att.predict_from_beta(att_extractor, "Rep_C", X_te_std, beta) * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
    del HtH, HtY, beta

    print("[5/5] Training and Evaluating Compact Linear Ridge Readout (Rep A, direct ridge)...")
    m_lin_cbf = ReadoutModel("linear", seed=42)
    HtH, HtY = m_lin_cbf.accumulate_HtH_HtY(cbf_extractor, "Rep_A", X_tr_std, Y_cbf_tr_std)
    beta = m_lin_cbf.solve_alpha(HtH, HtY, 10.0)
    pred_cbf_linear = np.clip(m_lin_cbf.predict_from_beta(cbf_extractor, "Rep_A", X_te_std, beta) * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
    del HtH, HtY, beta

    m_lin_att = ReadoutModel("linear", seed=42)
    HtH, HtY = m_lin_att.accumulate_HtH_HtY(att_extractor, "Rep_A", X_tr_std, Y_att_tr_std)
    beta = m_lin_att.solve_alpha(HtH, HtY, 10.0)
    pred_att_linear = np.clip(m_lin_att.predict_from_beta(att_extractor, "Rep_A", X_te_std, beta) * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
    del HtH, HtY, beta

    test_models = {
        "Baseline A (DNN-G)": (pred_cbf_dnn, pred_att_dnn),
        "Historical Feature ELM (Rep A, 512, a=0.01)": (pred_cbf_hist, pred_att_hist),
        "Penultimate Candidate (Rep A, 512, Tuned)": (pred_cbf_penult, pred_att_penult),
        "Multi-Layer Candidate (Rep C, 512, Tuned)": (pred_cbf_multi, pred_att_multi),
        "Compact Linear Ridge (Rep A, Direct Ridge)": (pred_cbf_linear, pred_att_linear)
    }

    # 3. Compute Metrics & Statistical Significance
    cbf_dnn_rmse = calc_metrics(Y_cbf_te, pred_cbf_dnn)["RMSE"]
    att_dnn_rmse = calc_metrics(Y_att_te, pred_att_dnn)["RMSE"]

    early_att_mask = (Y_att_te >= 0.5) & (Y_att_te < 1.0)

    test_metrics_rows = []
    print("\n--- Computing Locked Test Cohort Metrics & Significance ---")
    for name, (pcbf, patt) in test_models.items():
        m_cbf = calc_metrics(Y_cbf_te, pcbf)
        m_att = calc_metrics(Y_att_te, patt)
        m_early = calc_metrics(Y_att_te[early_att_mask], patt[early_att_mask])

        cbf_imp = (cbf_dnn_rmse - m_cbf["RMSE"]) / cbf_dnn_rmse * 100.0
        att_imp = (att_dnn_rmse - m_att["RMSE"]) / att_dnn_rmse * 100.0

        # Paired statistics vs Baseline DNN
        t_stat_cbf, p_val_cbf, w_stat_cbf, pw_cbf = paired_statistical_tests(Y_cbf_te, pred_cbf_dnn, pcbf)
        t_stat_att, p_val_att, w_stat_att, pw_att = paired_statistical_tests(Y_att_te, pred_att_dnn, patt)

        # Bootstrap 95% CI of RMSE difference vs Baseline DNN
        ci_cbf_low, ci_cbf_high = bootstrap_paired_diff(Y_cbf_te, pred_cbf_dnn, pcbf)
        ci_att_low, ci_att_high = bootstrap_paired_diff(Y_att_te, pred_att_dnn, patt)

        test_metrics_rows.append({
            "Model": name,
            "CBF_RMSE": round(m_cbf["RMSE"], 4),
            "CBF_MAE": round(m_cbf["MAE"], 4),
            "CBF_CCC": round(m_cbf["CCC"], 4),
            "CBF_R2": round(m_cbf["R2"], 4),
            "CBF_Bias": round(m_cbf["Bias"], 4),
            "CBF_Imp_%": round(cbf_imp, 2),
            "CBF_Diff_95CI": f"[{ci_cbf_low:+.4f}, {ci_cbf_high:+.4f}]",
            "CBF_p_val_paired_t": f"{p_val_cbf:.2e}",
            "ATT_RMSE": round(m_att["RMSE"], 4),
            "ATT_MAE": round(m_att["MAE"], 4),
            "ATT_CCC": round(m_att["CCC"], 4),
            "ATT_R2": round(m_att["R2"], 4),
            "ATT_Bias": round(m_att["Bias"], 4),
            "Early_ATT_RMSE": round(m_early["RMSE"], 4),
            "ATT_Imp_%": round(att_imp, 2),
            "ATT_Diff_95CI": f"[{ci_att_low:+.5f}, {ci_att_high:+.5f}]",
            "ATT_p_val_paired_t": f"{p_val_att:.2e}"
        })

    test_metrics_df = pd.DataFrame(test_metrics_rows)
    test_metrics_csv = os.path.join(OUT_DIR, "final_test_metrics.csv")
    test_metrics_df.to_csv(test_metrics_csv, index=False)
    print("\n--- Final Locked Test Results ---")
    print(test_metrics_df[["Model", "CBF_RMSE", "CBF_Imp_%", "ATT_RMSE", "Early_ATT_RMSE", "ATT_Imp_%", "CBF_Diff_95CI", "ATT_Diff_95CI"]].to_string())

    # 4. SNR-Stratified Test Performance
    print("\nComputing Locked Test SNR Breakdown...")
    unique_snrs = sorted(np.unique(SNR_te), key=lambda x: (np.isinf(x), x))
    snr_test_rows = []
    for snr in unique_snrs:
        mask = (SNR_te == snr) if not np.isinf(snr) else np.isinf(SNR_te)
        snr_label = f"{snr:.1f} dB" if not np.isinf(snr) else "Inf dB"

        for name, (pcbf, patt) in test_models.items():
            m_cbf = calc_metrics(Y_cbf_te[mask], pcbf[mask])
            m_att = calc_metrics(Y_att_te[mask], patt[mask])
            snr_test_rows.append({
                "SNR": snr_label,
                "Model": name,
                "CBF_RMSE": round(m_cbf["RMSE"], 4),
                "CBF_MAE": round(m_cbf["MAE"], 4),
                "CBF_CCC": round(m_cbf["CCC"], 4),
                "ATT_RMSE": round(m_att["RMSE"], 4),
                "ATT_MAE": round(m_att["MAE"], 4),
                "ATT_CCC": round(m_att["CCC"], 4)
            })

    snr_test_df = pd.DataFrame(snr_test_rows)
    snr_test_csv = os.path.join(OUT_DIR, "final_test_snr_stratified.csv")
    snr_test_df.to_csv(snr_test_csv, index=False)

    # 5. ATT Interval Breakdown
    print("\nComputing Locked Test ATT Interval Breakdown...")
    intervals = [
        ("[0.5, 1.0) s (Early)", 0.5, 1.0),
        ("[1.0, 1.5) s (Normal GM)", 1.0, 1.5),
        ("[1.5, 2.0) s (Intermediate)", 1.5, 2.0),
        ("[2.0, 2.5) s (Delayed)", 2.0, 2.5),
        ("[2.5, 3.0] s (Late)", 2.5, 3.001)
    ]
    att_test_rows = []
    for label, low, high in intervals:
        mask = (Y_att_te >= low) & (Y_att_te < high)
        n_samples = int(np.sum(mask))

        for name, (pcbf, patt) in test_models.items():
            m_att = calc_metrics(Y_att_te[mask], patt[mask])
            att_test_rows.append({
                "Interval": label,
                "N_Samples": n_samples,
                "Model": name,
                "ATT_RMSE": round(m_att["RMSE"], 4),
                "ATT_MAE": round(m_att["MAE"], 4),
                "ATT_Bias": round(m_att["Bias"], 4),
                "ATT_CCC": round(m_att["CCC"], 4)
            })

    att_test_df = pd.DataFrame(att_test_rows)
    att_test_csv = os.path.join(OUT_DIR, "final_test_att_intervals.csv")
    att_test_df.to_csv(att_test_csv, index=False)

    # 6. Final Test Publication Figures
    print("\nGenerating Locked Test Publication Figures in plots/...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    models_display = ["Baseline A (DNN-G)", "Historical Feature ELM (Rep A, 512, a=0.01)", "Multi-Layer Candidate (Rep C, 512, Tuned)", "Compact Linear Ridge (Rep A, Direct Ridge)"]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]

    sub_metrics = test_metrics_df[test_metrics_df["Model"].isin(models_display)]
    
    # CBF Bar Chart
    axes[0].bar(range(len(models_display)), sub_metrics["CBF_RMSE"], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_xticks(range(len(models_display)))
    axes[0].set_xticklabels(["DNN-G", "Hist ELM (Rep A)", "Multi-Layer (Rep C)", "Linear Ridge"], fontsize=10)
    axes[0].set_ylabel("CBF RMSE (mL / 100g / min)", fontsize=11)
    axes[0].set_title("Locked Test: CBF RMSE Across Estimators", fontsize=12, fontweight="bold")
    axes[0].set_ylim(3.68, 3.76)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(sub_metrics["CBF_RMSE"]):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    # ATT Bar Chart
    axes[1].bar(range(len(models_display)), sub_metrics["ATT_RMSE"], color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_xticks(range(len(models_display)))
    axes[1].set_xticklabels(["DNN-G", "Hist ELM (Rep A)", "Multi-Layer (Rep C)", "Linear Ridge"], fontsize=10)
    axes[1].set_ylabel("ATT RMSE (seconds)", fontsize=11)
    axes[1].set_title("Locked Test: ATT RMSE Across Estimators", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0.302, 0.306)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(sub_metrics["ATT_RMSE"]):
        axes[1].text(i, v + 0.0001, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "final_test_evaluation.png"), dpi=300)
    plt.close()

    print("\nStage 5 Locked-Test Evaluation complete. All files saved.")

if __name__ == "__main__":
    main()

