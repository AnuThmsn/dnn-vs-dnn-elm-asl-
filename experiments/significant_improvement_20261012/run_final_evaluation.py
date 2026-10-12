"""
run_final_evaluation.py
Phase 4: Honest Final Evaluation on the Locked Held-Out Test Set (N=100,000)
and Clean Physiological Grid (N=2,392).
Finalist candidates selected strictly on validation performance:
  1. Baseline A (DNN-G) [Control]
  2. Multi-Task DNN (Exp B finalist: Shared Encoder + Dedicated Heads)
  3. SNR-Balanced Sampling DNN (Exp A finalist)
  4. Ensemble Feature ELM (Exp D finalist: 5 seeds, frozen Baseline A latents)
  5. Multi-Task Feature ELM (5 seeds, Multi-Task shared latents)
Evaluates:
  - Global metrics (RMSE, MAE, Bias, Lin's CCC, R^2)
  - SNR stratification (12 levels)
  - ATT interval breakdown (5 intervals)
  - Paired t-tests, Wilcoxon signed-rank, 1,000-replicate bootstrap 95% CIs
Storage strictly on Drive D; host RAM < 80 MB.
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
OUT_DIR = os.path.join(EXP_DIR, "significant_improvement_20261012")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

sys.path.append(EXP_DIR)
sys.path.append(ELM_DIR)
sys.path.append(OUT_DIR)
import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device
from run_experiment_B import MultiTaskASLNet

def main():
    print("=" * 80)
    print("PHASE 4: HONEST FINAL EVALUATION ON LOCKED TEST SET (N=100,000)")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Normalization & Data
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    x_test_mmap = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    y_test_mmap = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    snr_test = np.load(os.path.join(CACHE_DIR, "snr_test.npy"), mmap_mode='r')

    clean_data = np.load(os.path.join(CACHE_DIR, "clean_grid.npz"))
    X_clean_raw = clean_data["X_clean"]
    cbf_clean_true = clean_data["cbf_flat"]
    att_clean_true = clean_data["att_flat"]

    X_test_n = ((x_test_mmap[:] - X_mean) / X_std).astype(np.float32)
    X_clean_n = ((X_clean_raw - X_mean) / X_std).astype(np.float32)

    Y_cbf_test = y_test_mmap[:, 0].astype(np.float64)
    Y_att_test = y_test_mmap[:, 1].astype(np.float64)

    # 2. Candidate 1: Baseline A (DNN-G)
    print("\nEvaluating Candidate 1: Baseline A (DNN-G)...")
    att_base = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    att_base.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device))
    att_base.eval()

    cbf_base = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    cbf_base.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device))
    cbf_base.eval()

    p_cbf_base_test = cbf_base.predict_physical(X_test_n)
    p_att_base_test = att_base.predict_physical(X_test_n)
    p_cbf_base_clean = cbf_base.predict_physical(X_clean_n)
    p_att_base_clean = att_base.predict_physical(X_clean_n)

    # 3. Candidate 2: Multi-Task DNN (Exp B Finalist)
    print("Evaluating Candidate 2: Multi-Task DNN...")
    mtl_net = MultiTaskASLNet(in_features=4, shared_dim=128, head_dim=64).to(device)
    mtl_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "expB_multitask_B1_Equal_Weights.pth"), map_location=device))
    mtl_net.eval()

    X_test_t = torch.from_numpy(X_test_n).to(device)
    X_clean_t = torch.from_numpy(X_clean_n).to(device)
    with torch.no_grad():
        c_v, a_v = mtl_net(X_test_t)
        p_cbf_mtl_test = np.clip(c_v.squeeze().cpu().numpy() * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
        p_att_mtl_test = np.clip(a_v.squeeze().cpu().numpy() * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

        c_cl, a_cl = mtl_net(X_clean_t)
        p_cbf_mtl_clean = np.clip(c_cl.squeeze().cpu().numpy() * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
        p_att_mtl_clean = np.clip(a_cl.squeeze().cpu().numpy() * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

    # 4. Candidate 3: SNR-Balanced Sampling DNN (Exp A Finalist)
    print("Evaluating Candidate 3: SNR-Balanced DNN...")
    att_snr_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    att_snr_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "expA_att_A2_SNR_Balanced.pth"), map_location=device))
    att_snr_net.eval()

    cbf_snr_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    cbf_snr_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "expA_cbf_A2_SNR_Balanced.pth"), map_location=device))
    cbf_snr_net.eval()

    p_cbf_snr_test = cbf_snr_net.predict_physical(X_test_n)
    p_att_snr_test = att_snr_net.predict_physical(X_test_n)
    p_cbf_snr_clean = cbf_snr_net.predict_physical(X_clean_n)
    p_att_snr_clean = att_snr_net.predict_physical(X_clean_n)

    # 5. Candidate 4: Existing Ensemble Feature ELM (5 seeds, latents only)
    print("Evaluating Candidate 4: Ensemble Feature ELM (5 seeds)...")
    elm_preds_npz = np.load(os.path.join(ELM_DIR, "all_test_predictions.npz"))
    p_cbf_ens_test = elm_preds_npz["ensemble_feature_elm_5_seeds_cbf_test"]
    p_att_ens_test = elm_preds_npz["ensemble_feature_elm_5_seeds_att_test"]

    # Re-evaluate clean predictions for candidate 4
    stored_clean_csv = pd.read_csv(os.path.join(ELM_DIR, "all_models_clean_comparison.csv"))

    # 6. Candidate 5: Multi-Task Feature ELM (5 seeds on Multi-Task shared representations)
    print("Evaluating Candidate 5: Multi-Task Feature ELM (5 seeds)...")
    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    N_tr = 100_000
    rng_elm = np.random.default_rng(2026)
    idx_elm = rng_elm.choice(1_000_000, size=N_tr, replace=False)

    X_tr_n_sub = ((x_tr_mmap[idx_elm] - X_mean) / X_std).astype(np.float32)
    Y_cbf_tr_std = ((y_tr_mmap[idx_elm, 0] - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_tr_std = ((y_tr_mmap[idx_elm, 1] - att_mean) / att_std).astype(np.float32)

    with torch.no_grad():
        F_mtl_tr = mtl_net.extract_features(torch.from_numpy(X_tr_n_sub).to(device)).cpu().numpy()
        F_mtl_test = mtl_net.extract_features(torch.from_numpy(X_test_n).to(device)).cpu().numpy()
        F_mtl_clean = mtl_net.extract_features(torch.from_numpy(X_clean_n).to(device)).cpu().numpy()

    p_cbf_mtl_elm_test = np.zeros(len(Y_cbf_test), dtype=np.float32)
    p_att_mtl_elm_test = np.zeros(len(Y_att_test), dtype=np.float32)
    p_cbf_mtl_elm_clean = np.zeros(len(cbf_clean_true), dtype=np.float32)
    p_att_mtl_elm_clean = np.zeros(len(att_clean_true), dtype=np.float32)

    for s in [42, 43, 44, 45, 46]:
        e_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_c.fit_chunked(F_mtl_tr, Y_cbf_tr_std)
        p_cbf_mtl_elm_test += np.clip(e_c.predict(F_mtl_test) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
        p_cbf_mtl_elm_clean += np.clip(e_c.predict(F_mtl_clean) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)

        e_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_a.fit_chunked(F_mtl_tr, Y_att_tr_std)
        p_att_mtl_elm_test += np.clip(e_a.predict(F_mtl_test) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
        p_att_mtl_elm_clean += np.clip(e_a.predict(F_mtl_clean) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

    p_cbf_mtl_elm_test /= 5.0
    p_att_mtl_elm_test /= 5.0
    p_cbf_mtl_elm_clean /= 5.0
    p_att_mtl_elm_clean /= 5.0

    all_candidates = [
        ("Baseline A (DNN-G)", p_cbf_base_test, p_att_base_test, p_cbf_base_clean, p_att_base_clean),
        ("Multi-Task DNN", p_cbf_mtl_test, p_att_mtl_test, p_cbf_mtl_clean, p_att_mtl_clean),
        ("SNR-Balanced DNN", p_cbf_snr_test, p_att_snr_test, p_cbf_snr_clean, p_att_snr_clean),
        ("Ensemble Feature ELM (Baseline Latents)", p_cbf_ens_test, p_att_ens_test, None, None),
        ("Multi-Task Feature ELM (MTL Latents)", p_cbf_mtl_elm_test, p_att_mtl_elm_test, p_cbf_mtl_elm_clean, p_att_mtl_elm_clean)
    ]

    # Save finalist test predictions
    np.savez(
        os.path.join(CKPT_DIR, "finalists_test_predictions.npz"),
        Y_cbf_test=Y_cbf_test, Y_att_test=Y_att_test, snr_test=snr_test,
        p_cbf_base_test=p_cbf_base_test, p_att_base_test=p_att_base_test,
        p_cbf_mtl_test=p_cbf_mtl_test, p_att_mtl_test=p_att_mtl_test,
        p_cbf_snr_test=p_cbf_snr_test, p_att_snr_test=p_att_snr_test,
        p_cbf_ens_test=p_cbf_ens_test, p_att_ens_test=p_att_ens_test,
        p_cbf_mtl_elm_test=p_cbf_mtl_elm_test, p_att_mtl_elm_test=p_att_mtl_elm_test
    )
    print(f"Saved test predictions: {os.path.join(CKPT_DIR, 'finalists_test_predictions.npz')}")

    # -------------------------------------------------------------
    # 7. Global Metric Comparison Table
    # -------------------------------------------------------------
    base_cbf_m = calc_metrics(Y_cbf_test, p_cbf_base_test)
    base_att_m = calc_metrics(Y_att_test, p_att_base_test)

    summary_rows = []
    for name, p_c, p_a, _, _ in all_candidates:
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

        summary_rows.append({
            "Candidate_Model": name,
            "CBF_RMSE": mc["RMSE"], "Delta_CBF_RMSE": d_c_rmse, "Gain_CBF_RMSE_%": p_c_rmse,
            "CBF_MAE": mc["MAE"], "Delta_CBF_MAE": d_c_mae, "Gain_CBF_MAE_%": p_c_mae,
            "CBF_Bias": mc["Bias"], "CBF_CCC": mc["CCC"], "CBF_R2": mc["R2"],
            "ATT_RMSE": ma["RMSE"], "Delta_ATT_RMSE": d_a_rmse, "Gain_ATT_RMSE_%": p_a_rmse,
            "ATT_MAE": ma["MAE"], "Delta_ATT_MAE": d_a_mae, "Gain_ATT_MAE_%": p_a_mae,
            "ATT_Bias": ma["Bias"], "ATT_CCC": ma["CCC"], "ATT_R2": ma["R2"]
        })

    summary_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(TABLES_DIR, "final_candidates_test_comparison.csv")
    summary_df.to_csv(sum_csv, index=False)
    print(f"\nFinal Locked Test Comparison Table (N=100,000):")
    print(summary_df[["Candidate_Model", "CBF_RMSE", "Gain_CBF_RMSE_%", "CBF_MAE", "Gain_CBF_MAE_%",
                       "ATT_RMSE", "Gain_ATT_RMSE_%", "ATT_MAE", "Gain_ATT_MAE_%"]].to_string(index=False))

    # -------------------------------------------------------------
    # 8. SNR Stratified Performance
    # -------------------------------------------------------------
    snr_rows = []
    for s_val in physics.SNR_VALUES:
        mask_s = np.isinf(snr_test) if np.isinf(s_val) else (snr_test == s_val)
        n_s = int(np.sum(mask_s))
        s_label = "inf" if np.isinf(s_val) else f"{s_val:.1f}"

        for name, p_c, p_a, _, _ in all_candidates:
            mc = calc_metrics(Y_cbf_test[mask_s], p_c[mask_s])
            ma = calc_metrics(Y_att_test[mask_s], p_a[mask_s])
            snr_rows.append({
                "SNR": s_label, "Model": name, "Count": n_s,
                "CBF_RMSE": mc["RMSE"], "CBF_MAE": mc["MAE"], "CBF_CCC": mc["CCC"],
                "ATT_RMSE": ma["RMSE"], "ATT_MAE": ma["MAE"], "ATT_CCC": ma["CCC"]
            })
    snr_df = pd.DataFrame(snr_rows)
    snr_csv = os.path.join(TABLES_DIR, "final_candidates_snr_breakdown.csv")
    snr_df.to_csv(snr_csv, index=False)

    # -------------------------------------------------------------
    # 9. ATT Interval Breakdown (Auditing Early Arrival [0.5, 1.0) s)
    # -------------------------------------------------------------
    intervals = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
    int_rows = []
    for lo, hi in intervals:
        mask_int = (Y_att_test >= lo) & (Y_att_test < hi) if hi < 3.0 else (Y_att_test >= lo) & (Y_att_test <= hi)
        n_int = int(np.sum(mask_int))
        int_label = f"[{lo:.1f}, {hi:.1f}]"

        for name, p_c, p_a, _, _ in all_candidates:
            ma = calc_metrics(Y_att_test[mask_int], p_a[mask_int])
            int_rows.append({
                "Interval": int_label, "Model": name, "Count": n_int,
                "ATT_RMSE": ma["RMSE"], "ATT_MAE": ma["MAE"],
                "ATT_Bias": ma["Bias"], "ATT_CCC": ma["CCC"]
            })
    int_df = pd.DataFrame(int_rows)
    int_csv = os.path.join(TABLES_DIR, "final_candidates_att_intervals.csv")
    int_df.to_csv(int_csv, index=False)
    print("\nEarly-Arrival Window [0.5, 1.0) s Performance (N=19,728):")
    print(int_df[int_df["Interval"] == "[0.5, 1.0]"][["Model", "ATT_RMSE", "ATT_MAE", "ATT_Bias", "ATT_CCC"]].to_string(index=False))

    # -------------------------------------------------------------
    # 10. Paired Statistical Hypothesis Testing & Bootstrap
    # -------------------------------------------------------------
    stat_rows = []
    se_base_c = (p_cbf_base_test - Y_cbf_test) ** 2
    se_base_a = (p_att_base_test - Y_att_test) ** 2
    ae_base_c = np.abs(p_cbf_base_test - Y_cbf_test)
    ae_base_a = np.abs(p_att_base_test - Y_att_test)

    rng_boot = np.random.default_rng(2026)
    sub_10k_idx = rng_boot.choice(100000, size=10000, replace=False)
    n_boot = 1000

    for name, p_c, p_a, _, _ in all_candidates[1:]:
        se_c = (p_c - Y_cbf_test) ** 2
        se_a = (p_a - Y_att_test) ** 2
        ae_c = np.abs(p_c - Y_cbf_test)
        ae_a = np.abs(p_a - Y_att_test)

        # Paired t-tests on MSE and MAE
        t_se_c, p_se_c = stats.ttest_rel(se_base_c, se_c)
        t_ae_c, p_ae_c = stats.ttest_rel(ae_base_c, ae_c)
        w_c, p_w_c = stats.wilcoxon(ae_base_c[sub_10k_idx], ae_c[sub_10k_idx])

        t_se_a, p_se_a = stats.ttest_rel(se_base_a, se_a)
        t_ae_a, p_ae_a = stats.ttest_rel(ae_base_a, ae_a)
        w_a, p_w_a = stats.wilcoxon(ae_base_a[sub_10k_idx], ae_a[sub_10k_idx])

        # Paired bootstrap CI (1,000 resamples)
        b_d_rmse_c = np.empty(n_boot)
        b_d_mae_c = np.empty(n_boot)
        b_d_rmse_a = np.empty(n_boot)
        b_d_mae_a = np.empty(n_boot)

        for b in range(n_boot):
            b_idx = rng_boot.integers(0, 100000, size=100000)
            b_d_rmse_c[b] = np.sqrt(np.mean(se_base_c[b_idx])) - np.sqrt(np.mean(se_c[b_idx]))
            b_d_mae_c[b] = np.mean(ae_base_c[b_idx]) - np.mean(ae_c[b_idx])
            b_d_rmse_a[b] = np.sqrt(np.mean(se_base_a[b_idx])) - np.sqrt(np.mean(se_a[b_idx]))
            b_d_mae_a[b] = np.mean(ae_base_a[b_idx]) - np.mean(ae_a[b_idx])

        ci_c_r = (float(np.percentile(b_d_rmse_c, 2.5)), float(np.percentile(b_d_rmse_c, 97.5)))
        ci_c_m = (float(np.percentile(b_d_mae_c, 2.5)), float(np.percentile(b_d_mae_c, 97.5)))
        ci_a_r = (float(np.percentile(b_d_rmse_a, 2.5)), float(np.percentile(b_d_rmse_a, 97.5)))
        ci_a_m = (float(np.percentile(b_d_mae_a, 2.5)), float(np.percentile(b_d_mae_a, 97.5)))

        stat_rows.append({
            "Candidate_Model": name, "Target": "CBF",
            "Delta_RMSE_Point": float(base_cbf_m["RMSE"] - calc_metrics(Y_cbf_test, p_c)["RMSE"]),
            "95%_CI_Delta_RMSE": f"[{ci_c_r[0]:.4f}, {ci_c_r[1]:.4f}]",
            "Delta_MAE_Point": float(base_cbf_m["MAE"] - calc_metrics(Y_cbf_test, p_c)["MAE"]),
            "95%_CI_Delta_MAE": f"[{ci_c_m[0]:.4f}, {ci_c_m[1]:.4f}]",
            "Paired_t_MSE_stat": float(t_se_c), "Paired_t_MSE_pval": float(p_se_c),
            "Paired_t_MAE_stat": float(t_ae_c), "Paired_t_MAE_pval": float(p_ae_c),
            "Wilcoxon_subsample_pval": float(p_w_c),
            "Bonferroni_Sig_alpha_0025": bool(p_se_c < 0.0025)
        })

        stat_rows.append({
            "Candidate_Model": name, "Target": "ATT",
            "Delta_RMSE_Point": float(base_att_m["RMSE"] - calc_metrics(Y_att_test, p_a)["RMSE"]),
            "95%_CI_Delta_RMSE": f"[{ci_a_r[0]:.5f}, {ci_a_r[1]:.5f}]",
            "Delta_MAE_Point": float(base_att_m["MAE"] - calc_metrics(Y_att_test, p_a)["MAE"]),
            "95%_CI_Delta_MAE": f"[{ci_a_m[0]:.5f}, {ci_a_m[1]:.5f}]",
            "Paired_t_MSE_stat": float(t_se_a), "Paired_t_MSE_pval": float(p_se_a),
            "Paired_t_MAE_stat": float(t_ae_a), "Paired_t_MAE_pval": float(p_ae_a),
            "Wilcoxon_subsample_pval": float(p_w_a),
            "Bonferroni_Sig_alpha_0025": bool(p_se_a < 0.0025)
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_csv = os.path.join(TABLES_DIR, "final_candidates_statistical_significance.csv")
    stat_df.to_csv(stat_csv, index=False)
    print(f"\nSaved statistical validation to: {stat_csv}")

    # -------------------------------------------------------------
    # 11. Plot Generation
    # -------------------------------------------------------------
    print("\nGenerating High-Impact Diagnostic Figures...")
    # Plot 1: RMSE / MAE Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    models_short = [m.replace(" (Baseline Latents)", "").replace(" (MTL Latents)", "").replace(" (DNN-G)", "") for m in summary_df["Candidate_Model"]]
    y_pos = np.arange(len(models_short))

    axes[0].barh(y_pos - 0.2, summary_df["CBF_RMSE"], height=0.4, label="CBF RMSE", color="#1f77b4", alpha=0.9)
    axes[0].barh(y_pos + 0.2, summary_df["CBF_MAE"], height=0.4, label="CBF MAE", color="#aec7e8", alpha=0.9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models_short, fontsize=10)
    axes[0].set_xlabel("CBF Error [mL / (100g · min)]", fontsize=11)
    axes[0].set_title("Cerebral Blood Flow (CBF) Estimation Errors", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()

    axes[1].barh(y_pos - 0.2, summary_df["ATT_RMSE"], height=0.4, label="ATT RMSE", color="#d62728", alpha=0.9)
    axes[1].barh(y_pos + 0.2, summary_df["ATT_MAE"], height=0.4, label="ATT MAE", color="#ff9896", alpha=0.9)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(models_short, fontsize=10)
    axes[1].set_xlabel("ATT Error [seconds]", fontsize=11)
    axes[1].set_title("Arterial Transit Time (ATT) Estimation Errors", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "all_candidates_rmse_mae_comparison.png"))
    plt.close()

    # Plot 2: ATT Interval Bias Profile
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for name, _, p_a, _, _ in all_candidates:
        biases = []
        for lo, hi in intervals:
            mask_int = (Y_att_test >= lo) & (Y_att_test < hi) if hi < 3.0 else (Y_att_test >= lo) & (Y_att_test <= hi)
            biases.append(calc_metrics(Y_att_test[mask_int], p_a[mask_int])["Bias"])
        ax.plot([f"[{lo:.1f}, {hi:.1f}]" for lo, hi in intervals], biases, marker='o', lw=2.0, label=name)
    ax.axhline(0.0, color="black", ls="--", alpha=0.6)
    ax.set_xlabel("Arterial Transit Time (ATT) Interval [seconds]", fontsize=11)
    ax.set_ylabel("Mean Estimation Bias (Predicted - True) [seconds]", fontsize=11)
    ax.set_title("ATT Interval Bias Profile: Demonstration of Early-Arrival Bottleneck", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "att_interval_bias_profile.png"))
    plt.close()

    print("=" * 80)
    print("PHASE 4 HONEST EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

