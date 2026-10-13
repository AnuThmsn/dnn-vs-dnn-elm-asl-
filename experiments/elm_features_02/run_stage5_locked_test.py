"""
run_stage5_locked_test.py
Stage 5: Final evaluation of selected Experiment 2 models on the locked test cohort (N=100,000).
Models compared:
  1. Baseline A DNN (Experiment G DNN-G)
  2. Historical Feature ELM (5 seeds, H=512, alpha=1e-2, tanh, Rep A)
  3. Experiment 1 Tuned ELM (5 seeds, H=2048, optimal alphas, Rep A)
  4. Experiment 2 Feature-Fusion ELM (5 seeds, Rep D: Latents + Scaled Physics features)
  5. Experiment 2 Out-Of-Fold Residual ELM (5 seeds, Rep A: Latents only)
  6. Experiment 2 Feature-Fusion Residual ELM (5 seeds, Rep D: Latents + Physics)
Evaluates:
  - Overall RMSE, MAE, Bias, Lin's CCC, R^2, and percentage gains
  - Early-ATT ([0.5, 1.0) s) diagnostic metrics
  - Stratified performance across all 12 SNR levels [5, 8, 10, 12, 15, 20, 30, 50, 100, 200, 500, inf]
  - ATT interval stratification across 5 clinical windows
  - Paired bootstrap 95% confidence intervals (B=1000) for RMSE differences
Saves:
  - final_test_metrics.csv
  - final_test_snr_stratified.csv
  - final_test_att_intervals.csv
  - final_test_predictions.npz
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
OUT_DIR = os.path.join(EXP_DIR, "elm_features_02")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.append(EXP_DIR)
sys.path.append(os.path.join(EXP_DIR, "elm_models"))
sys.path.append(OUT_DIR)

import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device
import feature_engineering as fe

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
    print(f"Compute Device: {device}")
    print("=" * 80)

    # 1. Normalization parameters & Checkpoints
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    att_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device))
    att_net.eval()

    cbf_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device))
    cbf_net.eval()

    # 2. Datasets
    # Training slice (100k, seed 2026) for fitting candidate models
    N_tr = 100_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_te_mmap = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    y_te_mmap = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    snr_te = np.load(os.path.join(CACHE_DIR, "SNR_test.npy"))

    X_tr_raw = x_tr_mmap[idx_tr]
    Y_cbf_tr = y_tr_mmap[idx_tr, 0].astype(np.float32)
    Y_att_tr = y_tr_mmap[idx_tr, 1].astype(np.float32)
    Y_cbf_tr_std = ((Y_cbf_tr - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_tr_std = ((Y_att_tr - att_mean) / att_std).astype(np.float32)
    X_tr_n = ((X_tr_raw - X_mean) / X_std).astype(np.float32)

    X_te_raw = x_te_mmap[:]
    X_te_n = ((X_te_raw - X_mean) / X_std).astype(np.float32)
    Y_cbf_te = y_te_mmap[:, 0].astype(np.float64)
    Y_att_te = y_te_mmap[:, 1].astype(np.float64)

    early_mask_te = (Y_att_te >= 0.5) & (Y_att_te < 1.0)

    # 3. Baseline A Test Predictions
    pred_cbf_dnn = cbf_net.predict_physical(X_te_n).astype(np.float64)
    pred_att_dnn = att_net.predict_physical(X_te_n).astype(np.float64)

    # 4. Extract Penultimate Latents (Rep A)
    print("\nExtracting Penultimate Latents for Train and Test...")
    with torch.no_grad():
        F_lat_cbf_tr = cbf_net.penultimate_features(torch.from_numpy(X_tr_n).to(device))
        F_lat_cbf_te = cbf_net.penultimate_features(torch.from_numpy(X_te_n).to(device))
        F_lat_att_tr = att_net.penultimate_features(torch.from_numpy(X_tr_n).to(device))
        F_lat_att_te = att_net.penultimate_features(torch.from_numpy(X_te_n).to(device))

    # 5. Extract Scaled Physics Features (Rep D)
    print("Extracting Scaled Physics Features for Train and Test...")
    F_phys_tr_raw = fe.extract_physics_features(X_tr_raw)
    F_phys_te_raw = fe.extract_physics_features(X_te_raw)
    mu_phys, std_phys = fe.fit_feature_scaler(F_phys_tr_raw)
    F_phys_tr = fe.apply_feature_scaler(F_phys_tr_raw, mu_phys, std_phys)
    F_phys_te = fe.apply_feature_scaler(F_phys_te_raw, mu_phys, std_phys)

    F_fusion_cbf_tr = np.hstack([F_lat_cbf_tr, F_phys_tr])
    F_fusion_cbf_te = np.hstack([F_lat_cbf_te, F_phys_te])
    F_fusion_att_tr = np.hstack([F_lat_att_tr, F_phys_tr])
    F_fusion_att_te = np.hstack([F_lat_att_te, F_phys_te])

    # 6. Load OOF Predictions & Residual Targets
    oof_cache_path = os.path.join(OUT_DIR, "oof_predictions_train.npz")
    if not os.path.exists(oof_cache_path):
        raise FileNotFoundError(f"Missing OOF predictions cache: {oof_cache_path}. Run Stage 3 first.")
    oof_data = np.load(oof_cache_path)
    oof_cbf_pred = oof_data["oof_cbf_pred"]
    oof_att_pred = oof_data["oof_att_pred"]

    res_cbf_tr = Y_cbf_tr - oof_cbf_pred
    res_att_tr = Y_att_tr - oof_att_pred
    res_cbf_std = float(np.std(res_cbf_tr)) + 1e-6
    res_att_std = float(np.std(res_att_tr)) + 1e-6
    res_cbf_tr_n = (res_cbf_tr / res_cbf_std).astype(np.float32)
    res_att_tr_n = (res_att_tr / res_att_std).astype(np.float32)

    # 7. Load Historical Predictions for Baseline Parity
    hist_preds_path = os.path.join(EXP_DIR, "elm_models", "all_test_predictions.npz")
    hist_data = np.load(hist_preds_path)
    pred_cbf_hist_elm5 = hist_data["ensemble_feature_elm_5_seeds_cbf_test"].astype(np.float64)
    pred_att_hist_elm5 = hist_data["ensemble_feature_elm_5_seeds_att_test"].astype(np.float64)

    # 8. Train and Predict Candidates on Test Cohort (5 seeds)
    seeds = [42, 43, 44, 45, 46]

    # --- Candidate 1: Experiment 1 Tuned ELM (Rep A, 5 seeds) ---
    print("\nFitting Model 3: Exp 1 Tuned ELM (Rep A, 5 seeds)...")
    cbf_m3, att_m3 = [], []
    for s in seeds:
        ec = GPUChunkedELM(n_hidden=2048, alpha=1.0, activation="elu", seed=s)
        ec.fit_chunked(F_lat_cbf_tr, Y_cbf_tr_std)
        cbf_m3.append(np.clip(ec.predict(F_lat_cbf_te) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX))

        ea = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=s)
        ea.fit_chunked(F_lat_att_tr, Y_att_tr_std)
        att_m3.append(np.clip(ea.predict(F_lat_att_te) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX))
    pred_cbf_tuned_repA = np.mean(cbf_m3, axis=0).astype(np.float64)
    pred_att_tuned_repA = np.mean(att_m3, axis=0).astype(np.float64)

    # --- Candidate 2: Experiment 2 Feature-Fusion ELM (Rep D, 5 seeds) ---
    print("Fitting Model 4: Exp 2 Feature-Fusion ELM (Rep D, 5 seeds)...")
    cbf_m4, att_m4 = [], []
    for s in seeds:
        ec = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="elu", seed=s)
        ec.fit_chunked(F_fusion_cbf_tr, Y_cbf_tr_std)
        cbf_m4.append(np.clip(ec.predict(F_fusion_cbf_te) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX))

        ea = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=s)
        ea.fit_chunked(F_fusion_att_tr, Y_att_tr_std)
        att_m4.append(np.clip(ea.predict(F_fusion_att_te) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX))
    pred_cbf_fusion_repD = np.mean(cbf_m4, axis=0).astype(np.float64)
    pred_att_fusion_repD = np.mean(att_m4, axis=0).astype(np.float64)

    # --- Candidate 3: Experiment 2 OOF Residual ELM (Rep A, 5 seeds, lambda=0.5) ---
    print("Fitting Model 5: Exp 2 OOF Residual ELM (Rep A, 5 seeds)...")
    cbf_m5, att_m5 = [], []
    for s in seeds:
        ec = GPUChunkedELM(n_hidden=2048, alpha=1.0, activation="tanh", seed=s)
        ec.fit_chunked(F_lat_cbf_tr, res_cbf_tr_n)
        cbf_m5.append(ec.predict(F_lat_cbf_te) * res_cbf_std)

        ea = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=s)
        ea.fit_chunked(F_lat_att_tr, res_att_tr_n)
        att_m5.append(ea.predict(F_lat_att_te) * res_att_std)
    res_cbf_pred_repA = np.mean(cbf_m5, axis=0)
    res_att_pred_repA = np.mean(att_m5, axis=0)
    pred_cbf_oof_res_repA = np.clip(pred_cbf_dnn + 0.5 * res_cbf_pred_repA, physics.CBF_MIN, physics.CBF_MAX).astype(np.float64)
    pred_att_oof_res_repA = np.clip(pred_att_dnn + 0.5 * res_att_pred_repA, physics.ATT_MIN, physics.ATT_MAX).astype(np.float64)

    # --- Candidate 4: Experiment 2 Feature-Fusion Residual ELM (Rep D, 5 seeds, lambda=0.5) ---
    print("Fitting Model 6: Exp 2 Feature-Fusion Residual ELM (Rep D, 5 seeds)...")
    cbf_m6, att_m6 = [], []
    for s in seeds:
        ec = GPUChunkedELM(n_hidden=2048, alpha=1.0, activation="tanh", seed=s)
        ec.fit_chunked(F_fusion_cbf_tr, res_cbf_tr_n)
        cbf_m6.append(ec.predict(F_fusion_cbf_te) * res_cbf_std)

        ea = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=s)
        ea.fit_chunked(F_fusion_att_tr, res_att_tr_n)
        att_m6.append(ea.predict(F_fusion_att_te) * res_att_std)
    res_cbf_pred_repD = np.mean(cbf_m6, axis=0)
    res_att_pred_repD = np.mean(att_m6, axis=0)
    pred_cbf_oof_res_repD = np.clip(pred_cbf_dnn + 0.5 * res_cbf_pred_repD, physics.CBF_MIN, physics.CBF_MAX).astype(np.float64)
    pred_att_oof_res_repD = np.clip(pred_att_dnn + 0.5 * res_att_pred_repD, physics.ATT_MIN, physics.ATT_MAX).astype(np.float64)

    # 9. Compile Full Evaluation Suite
    models = [
        ("Baseline A (DNN-G)", pred_cbf_dnn, pred_att_dnn, "StandardizedNet (Frozen)"),
        ("Historical Feature ELM (5 seeds)", pred_cbf_hist_elm5, pred_att_hist_elm5, "H=512, alpha=1e-2, tanh, Rep A (5 seeds)"),
        ("Exp 1 Tuned ELM (5 seeds)", pred_cbf_tuned_repA, pred_att_tuned_repA, "H=2048, tuned alpha, Rep A (5 seeds)"),
        ("Exp 2 Feature-Fusion ELM (5 seeds)", pred_cbf_fusion_repD, pred_att_fusion_repD, "H=2048, alpha=100, Rep D (Latents + Physics)"),
        ("Exp 2 OOF Residual ELM (5 seeds, lambda=0.5)", pred_cbf_oof_res_repA, pred_att_oof_res_repA, "Rep A Residual correction (lambda=0.5)"),
        ("Exp 2 Fusion Residual ELM (5 seeds, lambda=0.5)", pred_cbf_oof_res_repD, pred_att_oof_res_repD, "Rep D Residual correction (lambda=0.5)")
    ]

    base_cbf_rmse = calc_metrics(Y_cbf_te, pred_cbf_dnn)["RMSE"]
    base_att_rmse = calc_metrics(Y_att_te, pred_att_dnn)["RMSE"]
    hist_cbf_rmse = calc_metrics(Y_cbf_te, pred_cbf_hist_elm5)["RMSE"]
    hist_att_rmse = calc_metrics(Y_att_te, pred_att_hist_elm5)["RMSE"]

    records = []
    print("\nComputing Locked Test Metrics & Bootstrap CIs...")
    for name, p_cbf, p_att, cfg_desc in models:
        m_c = calc_metrics(Y_cbf_te, p_cbf)
        m_a = calc_metrics(Y_att_te, p_att)
        m_early = calc_metrics(Y_att_te[early_mask_te], p_att[early_mask_te])

        # Paired bootstrap CI for RMSE difference vs Baseline DNN
        if name == "Baseline A (DNN-G)":
            cbf_ci = (0.0, 0.0)
            att_ci = (0.0, 0.0)
        else:
            cbf_ci = paired_bootstrap_ci(Y_cbf_te, pred_cbf_dnn, p_cbf, B=1000, seed=42)
            att_ci = paired_bootstrap_ci(Y_att_te, pred_att_dnn, p_att, B=1000, seed=42)

        records.append({
            "Model": name,
            "Configuration": cfg_desc,
            "CBF_RMSE": m_c["RMSE"],
            "CBF_MAE": m_c["MAE"],
            "CBF_Bias": m_c["Bias"],
            "CBF_CCC": m_c["CCC"],
            "CBF_R2": m_c["R2"],
            "CBF_Gain_vs_DNN_%": 100.0 * (base_cbf_rmse - m_c["RMSE"]) / base_cbf_rmse,
            "CBF_Gain_vs_HistELM_%": 100.0 * (hist_cbf_rmse - m_c["RMSE"]) / hist_cbf_rmse,
            "CBF_Delta_95CI_Low": cbf_ci[0],
            "CBF_Delta_95CI_High": cbf_ci[1],
            "ATT_RMSE": m_a["RMSE"],
            "ATT_MAE": m_a["MAE"],
            "ATT_Bias": m_a["Bias"],
            "ATT_CCC": m_a["CCC"],
            "ATT_R2": m_a["R2"],
            "ATT_Gain_vs_DNN_%": 100.0 * (base_att_rmse - m_a["RMSE"]) / base_att_rmse,
            "ATT_Gain_vs_HistELM_%": 100.0 * (hist_att_rmse - m_a["RMSE"]) / hist_att_rmse,
            "ATT_Delta_95CI_Low": att_ci[0],
            "ATT_Delta_95CI_High": att_ci[1],
            "Early_ATT_RMSE": m_early["RMSE"],
            "Early_ATT_Bias": m_early["Bias"],
            "Early_ATT_CCC": m_early["CCC"]
        })

    df_test = pd.DataFrame(records)
    out_test_csv = os.path.join(OUT_DIR, "final_test_metrics.csv")
    df_test.to_csv(out_test_csv, index=False)
    print(f"\nSaved Final Locked Test Metrics to {out_test_csv}:")
    print(df_test[["Model", "CBF_RMSE", "CBF_Gain_vs_DNN_%", "ATT_RMSE", "ATT_Gain_vs_DNN_%", "Early_ATT_RMSE"]].to_string(index=False))

    # 10. Stratified SNR Breakdown Across All 12 Actual Test SNRs
    snr_levels = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, np.inf]
    snr_records = []
    for snr_val in snr_levels:
        mask = np.isinf(snr_te) if np.isinf(snr_val) else (np.abs(snr_te - snr_val) < 1e-3)
        snr_lbl = "inf" if np.isinf(snr_val) else f"{int(snr_val)}"

        for name, p_cbf, p_att, _ in models:
            m_c = calc_metrics(Y_cbf_te[mask], p_cbf[mask])
            m_a = calc_metrics(Y_att_te[mask], p_att[mask])
            snr_records.append({
                "SNR": snr_lbl, "Model": name,
                "CBF_RMSE": m_c["RMSE"], "CBF_MAE": m_c["MAE"], "CBF_CCC": m_c["CCC"],
                "ATT_RMSE": m_a["RMSE"], "ATT_MAE": m_a["MAE"], "ATT_CCC": m_a["CCC"]
            })

    df_snr = pd.DataFrame(snr_records)
    snr_csv = os.path.join(OUT_DIR, "final_test_snr_stratified.csv")
    df_snr.to_csv(snr_csv, index=False)
    print(f"Saved SNR Stratified Test Metrics to {snr_csv}.")

    # 11. ATT Interval Stratification Across Clinical Windows
    att_windows = [
        ("[0.5, 1.0)", (Y_att_te >= 0.5) & (Y_att_te < 1.0)),
        ("[1.0, 1.5)", (Y_att_te >= 1.0) & (Y_att_te < 1.5)),
        ("[1.5, 2.0)", (Y_att_te >= 1.5) & (Y_att_te < 2.0)),
        ("[2.0, 2.5)", (Y_att_te >= 2.0) & (Y_att_te < 2.5)),
        ("[2.5, 3.0]", (Y_att_te >= 2.5) & (Y_att_te <= 3.0))
    ]
    interval_records = []
    for win_lbl, mask in att_windows:
        for name, p_cbf, p_att, _ in models:
            m_c = calc_metrics(Y_cbf_te[mask], p_cbf[mask])
            m_a = calc_metrics(Y_att_te[mask], p_att[mask])
            interval_records.append({
                "ATT_Window": win_lbl, "Model": name,
                "Sample_Count": int(np.sum(mask)),
                "CBF_RMSE": m_c["RMSE"], "CBF_MAE": m_c["MAE"], "CBF_CCC": m_c["CCC"],
                "ATT_RMSE": m_a["RMSE"], "ATT_MAE": m_a["MAE"], "ATT_Bias": m_a["Bias"], "ATT_CCC": m_a["CCC"]
            })

    df_intervals = pd.DataFrame(interval_records)
    int_csv = os.path.join(OUT_DIR, "final_test_att_intervals.csv")
    df_intervals.to_csv(int_csv, index=False)
    print(f"Saved ATT Interval Stratified Metrics to {int_csv}.")

    # 12. Save Predictions Array for Auditing Provenance
    pred_save_path = os.path.join(OUT_DIR, "final_test_predictions.npz")
    np.savez_compressed(
        pred_save_path,
        cbf_fusion_repD=pred_cbf_fusion_repD.astype(np.float32),
        att_fusion_repD=pred_att_fusion_repD.astype(np.float32),
        cbf_oof_res_repA=pred_cbf_oof_res_repA.astype(np.float32),
        att_oof_res_repA=pred_att_oof_res_repA.astype(np.float32),
        cbf_oof_res_repD=pred_cbf_oof_res_repD.astype(np.float32),
        att_oof_res_repD=pred_att_oof_res_repD.astype(np.float32)
    )
    print(f"Saved test predictions array to {pred_save_path}.")
    print("=" * 80)
    print("STAGE 5 LOCKED TEST EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()

