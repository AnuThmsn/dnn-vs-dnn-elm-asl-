"""
run_stage2_benchmark.py
Stage 2: Reproduce Fair Final Benchmark on Locked Test Cohort (N=100,000).
Evaluates Baseline A (DNN-G) vs. Best Defensible Feature ELM Candidates.
Computes:
- RMSE, MAE, Bias, R2, CCC
- Absolute difference (DNN - Cand) and Percentage Gain (%)
- Paired 95% bootstrap confidence intervals (B=2000 resamples)
- Paired t-test and Wilcoxon signed-rank test
- SNR-stratified metrics across all 12 levels
- ATT-interval stratified metrics
"""

import os
import sys
import gc
import time
import json
import numpy as np
import torch
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments", "elm_models"))

import memory_config
from physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, SNR_VALUES
)
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

OUT_DIR = os.path.join(BASE_DIR, "experiments", "final_results_05")
CACHE_DIR = os.path.join(BASE_DIR, "experiments", "cache")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[Stage 2 Benchmark] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def compute_bootstrap_paired_ci(y_true, pred_base, pred_cand, B=2000, seed=42):
    """Compute 95% paired bootstrap CI for (RMSE_base - RMSE_cand). Positive = candidate improvement."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        yt_b = y_true[idx]
        rmse_b = np.sqrt(np.mean((pred_base[idx] - yt_b) ** 2))
        rmse_c = np.sqrt(np.mean((pred_cand[idx] - yt_b) ** 2))
        diffs[b] = rmse_b - rmse_c  # Positive means candidate is better
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def compute_statistical_tests(y_true, pred_base, pred_cand, seed=42):
    """Paired t-test on squared errors and Wilcoxon on absolute errors (10,000 subsample)."""
    err_b_sq = (pred_base - y_true) ** 2
    err_c_sq = (pred_cand - y_true) ** 2
    t_stat, p_val_t = stats.ttest_rel(err_b_sq, err_c_sq)  # positive t means base has larger error

    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(len(y_true), size=min(10_000, len(y_true)), replace=False)
    abs_b = np.abs(pred_base[sub_idx] - y_true[sub_idx])
    abs_c = np.abs(pred_cand[sub_idx] - y_true[sub_idx])
    w_stat, p_val_w = stats.wilcoxon(abs_b, abs_c)
    return float(t_stat), float(p_val_t), float(w_stat), float(p_val_w)

def extract_penultimate_features(net, X_std, chunk_size=25_000):
    N = X_std.shape[0]
    dummy = torch.zeros((1, 4), device=device)
    with torch.no_grad():
        feat_dim = net.penultimate_features(dummy).shape[1]
    feats = np.empty((N, feat_dim), dtype=np.float32)
    n_chunks = int(np.ceil(N / chunk_size))
    with torch.no_grad():
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)
            xc = torch.from_numpy(X_std[start:end].astype(np.float32)).to(device)
            hc = net.penultimate_features(xc)
            if isinstance(hc, torch.Tensor):
                feats[start:end] = hc.cpu().numpy()
            else:
                feats[start:end] = hc
            del xc, hc
    return feats

def train_predict_elm(F_tr, Y_tr_std, F_te, y_std, y_mean, y_min, y_max,
                       n_hidden=512, alpha=1e-2, activation="tanh", seeds=[42]):
    preds_acc = np.zeros(F_te.shape[0], dtype=np.float64)
    for s in seeds:
        elm = GPUChunkedELM(n_hidden=n_hidden, alpha=alpha, activation=activation, seed=s)
        elm.fit_chunked(F_tr, Y_tr_std, chunk_size=25_000)
        p_std = elm.predict(F_te, chunk_size=25_000)
        p_phys = np.clip(p_std * y_std + y_mean, y_min, y_max)
        preds_acc += p_phys
        del elm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return preds_acc / len(seeds)

def main():
    print("=" * 80)
    print("STAGE 2: REPRODUCE FAIR FINAL BENCHMARK ON LOCKED TEST COHORT")
    print("=" * 80)

    # 1. Normalization & Models
    norm_path = os.path.join(BASE_DIR, "experiments", "baseline_A", "normalization_stats.npz")
    norm_data = np.load(norm_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    cbf_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, CBF_MIN, CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "experiments", "baseline_A", "CBFnet_G_baselineA.pth"), map_location=device, weights_only=False))
    cbf_net.eval()

    att_net = StandardizedNet(4, 4, 64, att_mean, att_std, ATT_MIN, ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "experiments", "baseline_A", "ATTnet_G_baselineA.pth"), map_location=device, weights_only=False))
    att_net.eval()

    # 2. Datasets
    print("Loading dataset arrays...")
    X_tr_raw = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    Y_tr_raw = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    X_te_raw = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    Y_te_raw = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    SNR_te = np.load(os.path.join(CACHE_DIR, "SNR_test.npy"))

    X_tr_std = (X_tr_raw[:] - X_mean) / X_std
    X_te_std = (X_te_raw[:] - X_mean) / X_std

    Y_cbf_tr_std = (Y_tr_raw[:, 0] - cbf_mean) / cbf_std
    Y_att_tr_std = (Y_tr_raw[:, 1] - att_mean) / att_std

    Y_cbf_te = Y_te_raw[:, 0].astype(np.float64)
    Y_att_te = Y_te_raw[:, 1].astype(np.float64)

    # 3. Model Predictions on Locked Test Cohort
    print("\n[1/4] Predicting Baseline A DNN-G...")
    pred_cbf_dnn = cbf_net.predict_physical(X_te_std)
    pred_att_dnn = att_net.predict_physical(X_te_std)

    print("[2/4] Extracting penultimate features from frozen DNN...")
    F_cbf_tr = extract_penultimate_features(cbf_net, X_tr_std)
    F_cbf_te = extract_penultimate_features(cbf_net, X_te_std)
    F_att_tr = extract_penultimate_features(att_net, X_tr_std)
    F_att_te = extract_penultimate_features(att_net, X_te_std)

    print("[3/4] Training and Predicting Single-Seed Feature ELM (H=512, alpha=1e-2, seed 42)...")
    pred_cbf_elm1 = train_predict_elm(F_cbf_tr, Y_cbf_tr_std, F_cbf_te, cbf_std, cbf_mean, CBF_MIN, CBF_MAX,
                                      n_hidden=512, alpha=1e-2, activation="tanh", seeds=[42])
    pred_att_elm1 = train_predict_elm(F_att_tr, Y_att_tr_std, F_att_te, att_std, att_mean, ATT_MIN, ATT_MAX,
                                      n_hidden=512, alpha=1e-2, activation="tanh", seeds=[42])

    print("[4/4] Training and Predicting 5-Seed Ensemble Feature ELM (H=512, alpha=1e-2, seeds 42-46)...")
    pred_cbf_elm5 = train_predict_elm(F_cbf_tr, Y_cbf_tr_std, F_cbf_te, cbf_std, cbf_mean, CBF_MIN, CBF_MAX,
                                      n_hidden=512, alpha=1e-2, activation="tanh", seeds=[42, 43, 44, 45, 46])
    pred_att_elm5 = train_predict_elm(F_att_tr, Y_att_tr_std, F_att_te, att_std, att_mean, ATT_MIN, ATT_MAX,
                                      n_hidden=512, alpha=1e-2, activation="tanh", seeds=[42, 43, 44, 45, 46])

    benchmark_models = {
        "Baseline A (DNN-G)": (pred_cbf_dnn, pred_att_dnn),
        "Single-Seed Feature ELM (Seed 42)": (pred_cbf_elm1, pred_att_elm1),
        "Ensemble Feature ELM (5 seeds)": (pred_cbf_elm5, pred_att_elm5)
    }

    # 4. Overall Benchmark Metrics Table
    cbf_dnn_rmse = calc_metrics(Y_cbf_te, pred_cbf_dnn)["RMSE"]
    att_dnn_rmse = calc_metrics(Y_att_te, pred_att_dnn)["RMSE"]
    early_mask = (Y_att_te >= 0.5) & (Y_att_te < 1.0)

    rows = []
    print("\n--- Computing Benchmark Metrics and Bootstrap CIs ---")
    for name, (pcbf, patt) in benchmark_models.items():
        m_cbf = calc_metrics(Y_cbf_te, pcbf)
        m_att = calc_metrics(Y_att_te, patt)
        m_early = calc_metrics(Y_att_te[early_mask], patt[early_mask])

        # Improvements relative to DNN (Positive = Better)
        cbf_delta_rmse = cbf_dnn_rmse - m_cbf["RMSE"]
        cbf_pct_gain = (cbf_delta_rmse / cbf_dnn_rmse) * 100.0

        att_delta_rmse = att_dnn_rmse - m_att["RMSE"]
        att_pct_gain = (att_delta_rmse / att_dnn_rmse) * 100.0

        if name == "Baseline A (DNN-G)":
            cbf_ci_low, cbf_ci_high = 0.0, 0.0
            cbf_p_t = np.nan
            att_ci_low, att_ci_high = 0.0, 0.0
            att_p_t = np.nan
        else:
            cbf_ci_low, cbf_ci_high = compute_bootstrap_paired_ci(Y_cbf_te, pred_cbf_dnn, pcbf)
            _, cbf_p_t, _, _ = compute_statistical_tests(Y_cbf_te, pred_cbf_dnn, pcbf)
            att_ci_low, att_ci_high = compute_bootstrap_paired_ci(Y_att_te, pred_att_dnn, patt)
            _, att_p_t, _, _ = compute_statistical_tests(Y_att_te, pred_att_dnn, patt)

        rows.append({
            "Model": name,
            "CBF_RMSE": round(m_cbf["RMSE"], 4),
            "CBF_MAE": round(m_cbf["MAE"], 4),
            "CBF_Bias": round(m_cbf["Bias"], 4),
            "CBF_CCC": round(m_cbf["CCC"], 4),
            "CBF_R2": round(m_cbf["R2"], 4),
            "CBF_Delta_RMSE": round(cbf_delta_rmse, 4),
            "CBF_Pct_Gain": round(cbf_pct_gain, 2),
            "CBF_Paired_95CI": f"[{cbf_ci_low:+.4f}, {cbf_ci_high:+.4f}]",
            "CBF_p_val": f"{cbf_p_t:.2e}" if not np.isnan(cbf_p_t) else "N/A",
            "ATT_RMSE": round(m_att["RMSE"], 4),
            "ATT_MAE": round(m_att["MAE"], 4),
            "ATT_Bias": round(m_att["Bias"], 4),
            "ATT_CCC": round(m_att["CCC"], 4),
            "ATT_R2": round(m_att["R2"], 4),
            "Early_ATT_RMSE": round(m_early["RMSE"], 4),
            "Early_ATT_Bias": round(m_early["Bias"], 4),
            "ATT_Delta_RMSE": round(att_delta_rmse, 4),
            "ATT_Pct_Gain": round(att_pct_gain, 2),
            "ATT_Paired_95CI": f"[{att_ci_low:+.5f}, {att_ci_high:+.5f}]",
            "ATT_p_val": f"{att_p_t:.2e}" if not np.isnan(att_p_t) else "N/A"
        })

    benchmark_df = pd.DataFrame(rows)
    benchmark_csv = os.path.join(OUT_DIR, "benchmark_metrics.csv")
    benchmark_df.to_csv(benchmark_csv, index=False)
    print("\n--- Benchmark Metrics Summary ---")
    print(benchmark_df[["Model", "CBF_RMSE", "CBF_Pct_Gain", "CBF_Paired_95CI", "ATT_RMSE", "ATT_Pct_Gain", "ATT_Paired_95CI", "Early_ATT_RMSE"]].to_string())

    # 5. Stratified Benchmark Metrics (SNR & ATT Intervals)
    print("\nComputing Stratified Benchmark Metrics...")
    strat_rows = []

    # Part A: SNR Stratification
    unique_snrs = sorted(np.unique(SNR_te), key=lambda x: (np.isinf(x), x))
    for snr in unique_snrs:
        mask = (SNR_te == snr) if not np.isinf(snr) else np.isinf(SNR_te)
        snr_label = f"{snr:.1f} dB" if not np.isinf(snr) else "Inf dB"
        n_samples = int(np.sum(mask))

        for name, (pcbf, patt) in benchmark_models.items():
            m_cbf = calc_metrics(Y_cbf_te[mask], pcbf[mask])
            m_att = calc_metrics(Y_att_te[mask], patt[mask])
            strat_rows.append({
                "Stratification_Type": "SNR",
                "Subgroup": snr_label,
                "N_Samples": n_samples,
                "Model": name,
                "CBF_RMSE": round(m_cbf["RMSE"], 4),
                "CBF_MAE": round(m_cbf["MAE"], 4),
                "CBF_CCC": round(m_cbf["CCC"], 4),
                "ATT_RMSE": round(m_att["RMSE"], 4),
                "ATT_MAE": round(m_att["MAE"], 4),
                "ATT_CCC": round(m_att["CCC"], 4)
            })

    # Part B: ATT Interval Stratification
    intervals = [
        ("[0.5, 1.0) s (Early)", 0.5, 1.0),
        ("[1.0, 1.5) s (Normal GM)", 1.0, 1.5),
        ("[1.5, 2.0) s (Intermediate)", 1.5, 2.0),
        ("[2.0, 2.5) s (Delayed)", 2.0, 2.5),
        ("[2.5, 3.0] s (Late)", 2.5, 3.001)
    ]
    for label, low, high in intervals:
        mask = (Y_att_te >= low) & (Y_att_te < high)
        n_samples = int(np.sum(mask))

        for name, (pcbf, patt) in benchmark_models.items():
            m_att = calc_metrics(Y_att_te[mask], patt[mask])
            m_cbf = calc_metrics(Y_cbf_te[mask], pcbf[mask])
            strat_rows.append({
                "Stratification_Type": "ATT_Interval",
                "Subgroup": label,
                "N_Samples": n_samples,
                "Model": name,
                "CBF_RMSE": round(m_cbf["RMSE"], 4),
                "CBF_MAE": round(m_cbf["MAE"], 4),
                "CBF_CCC": round(m_cbf["CCC"], 4),
                "ATT_RMSE": round(m_att["RMSE"], 4),
                "ATT_MAE": round(m_att["MAE"], 4),
                "ATT_CCC": round(m_att["CCC"], 4)
            })

    strat_df = pd.DataFrame(strat_rows)
    strat_csv = os.path.join(OUT_DIR, "benchmark_stratified_metrics.csv")
    strat_df.to_csv(strat_csv, index=False)
    print(f"Stratified metrics saved to {strat_csv}")

if __name__ == "__main__":
    main()

