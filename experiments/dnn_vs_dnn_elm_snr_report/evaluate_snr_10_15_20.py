"""
evaluate_snr_10_15_20.py
Phase 2-4: Strict Evaluation of Baseline DNN vs DNN-ELM at SNR 10, 15, and 20.
Calculates RMSE, MAE, and Lin's CCC with exact mathematical formulas.
Evaluates individual seeds (42..46), mean ± std across seeds, and the 5-seed ensemble.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

import physics
from train_baseline_A import StandardizedNet
from elm_models.elm_engine import GPUChunkedELM

OUT_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(REPO_ROOT, "experiments", "cache")
BASELINE_DIR = os.path.join(REPO_ROOT, "experiments", "baseline_A")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_exact_metrics(y_true, y_pred):
    """
    Computes exact RMSE, MAE, and Lin's Concordance Correlation Coefficient (CCC)
    using mathematically consistent population moments.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    diff = y_pred - y_true
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)

    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_t = np.mean((y_true - mean_t) ** 2)
    var_p = np.mean((y_pred - mean_p) ** 2)
    cov_tp = np.mean((y_true - mean_t) * (y_pred - mean_p))

    denom = var_t + var_p + (mean_t - mean_p) ** 2
    ccc = (2.0 * cov_tp) / denom if denom > 1e-12 else 0.0

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "CCC": float(ccc),
        "Bias": float(bias)
    }

def run_evaluation():
    print("=" * 80)
    print("PHASE 2-4: EVALUATION AT SNR 10, 15, AND 20")
    print("=" * 80)

    # 1. Load Normalization and Checkpoints
    norm_path = os.path.join(BASELINE_DIR, "normalization_stats.npz")
    norm_data = np.load(norm_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    cbf_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    att_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASELINE_DIR, "CBFnet_G_baselineA.pth"), map_location=device, weights_only=False))
    att_net.load_state_dict(torch.load(os.path.join(BASELINE_DIR, "ATTnet_G_baselineA.pth"), map_location=device, weights_only=False))
    cbf_net.eval(); att_net.eval()

    # 2. Load Locked Test Set
    X_test_raw = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')[:]
    Y_test_raw = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')[:]
    snr_test = np.load(os.path.join(CACHE_DIR, "snr_test.npy"), mmap_mode='r')[:]

    X_test_n = ((X_test_raw - X_mean) / X_std).astype(np.float32)

    # 3. Predict Baseline DNN
    print("\n[1/3] Predicting Baseline A DNN-G on Test Cohort...")
    pred_cbf_dnn = cbf_net.predict_physical(X_test_n)
    pred_att_dnn = att_net.predict_physical(X_test_n)

    # 4. Extract Penultimate Features and Predict Feature ELMs
    print("[2/3] Extracting Penultimate Features and Training ELMs...")
    X_tr_raw = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')[:]
    Y_tr_raw = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')[:]
    X_tr_n = ((X_tr_raw - X_mean) / X_std).astype(np.float32)

    def extract_feats(net, X_in):
        N = X_in.shape[0]
        dummy = torch.zeros((1, 4), device=device)
        feat_dim = net.penultimate_features(dummy).shape[1]
        feats = np.empty((N, feat_dim), dtype=np.float32)
        chunk = 50_000
        with torch.no_grad():
            for i in range(0, N, chunk):
                xc = torch.from_numpy(X_in[i:i+chunk]).to(device)
                h = net.penultimate_features(xc)
                feats[i:i+chunk] = h.cpu().numpy() if isinstance(h, torch.Tensor) else h
                del xc
        return feats

    F_cbf_tr = extract_feats(cbf_net, X_tr_n)
    F_cbf_te = extract_feats(cbf_net, X_test_n)
    F_att_tr = extract_feats(att_net, X_tr_n)
    F_att_te = extract_feats(att_net, X_test_n)

    Y_cbf_tr_std = (Y_tr_raw[:, 0] - cbf_mean) / cbf_std
    Y_att_tr_std = (Y_tr_raw[:, 1] - att_mean) / att_std

    # Train individual seeds
    seeds = [42, 43, 44, 45, 46]
    cbf_elm_seeds_preds = {}
    att_elm_seeds_preds = {}

    for s in seeds:
        print(f"  Fitting Feature ELM with Seed {s}...")
        elm_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, activation="tanh", seed=s)
        elm_c.fit_chunked(F_cbf_tr, Y_cbf_tr_std, chunk_size=50_000)
        p_c = elm_c.predict(F_cbf_te, chunk_size=50_000)
        cbf_elm_seeds_preds[s] = np.clip(p_c * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)

        elm_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, activation="tanh", seed=s)
        elm_a.fit_chunked(F_att_tr, Y_att_tr_std, chunk_size=50_000)
        p_a = elm_a.predict(F_att_te, chunk_size=50_000)
        att_elm_seeds_preds[s] = np.clip(p_a * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
        del elm_c, elm_a

    # 5-Seed Ensemble Prediction
    pred_cbf_elm_ens = np.mean([cbf_elm_seeds_preds[s] for s in seeds], axis=0)
    pred_att_elm_ens = np.mean([att_elm_seeds_preds[s] for s in seeds], axis=0)

    # 5. Evaluate at Requested SNR Levels (10, 15, 20)
    print("\n[3/3] Evaluating Models at SNR 10, 15, and 20...")
    target_snrs = [10.0, 15.0, 20.0]
    comparison_rows = []
    seed_detail_rows = []

    for s_val in target_snrs:
        mask = (snr_test == s_val)
        n_samples = int(np.sum(mask))

        y_true_cbf = Y_test_raw[mask, 0]
        y_true_att = Y_test_raw[mask, 1]

        # Baseline DNN
        m_dnn_c = calculate_exact_metrics(y_true_cbf, pred_cbf_dnn[mask])
        m_dnn_a = calculate_exact_metrics(y_true_att, pred_att_dnn[mask])

        # Ensemble ELM (5 seeds)
        m_elm_ens_c = calculate_exact_metrics(y_true_cbf, pred_cbf_elm_ens[mask])
        m_elm_ens_a = calculate_exact_metrics(y_true_att, pred_att_elm_ens[mask])

        # Single-Seed ELM (Seed 42)
        m_elm_s42_c = calculate_exact_metrics(y_true_cbf, cbf_elm_seeds_preds[42][mask])
        m_elm_s42_a = calculate_exact_metrics(y_true_att, att_elm_seeds_preds[42][mask])

        # Evaluate all individual seeds for standard deviation
        seed_cbf_rmses = []
        seed_cbf_maes = []
        seed_cbf_cccs = []
        seed_att_rmses = []
        seed_att_maes = []
        seed_att_cccs = []

        for s in seeds:
            mc = calculate_exact_metrics(y_true_cbf, cbf_elm_seeds_preds[s][mask])
            ma = calculate_exact_metrics(y_true_att, att_elm_seeds_preds[s][mask])
            seed_cbf_rmses.append(mc["RMSE"])
            seed_cbf_maes.append(mc["MAE"])
            seed_cbf_cccs.append(mc["CCC"])
            seed_att_rmses.append(ma["RMSE"])
            seed_att_maes.append(ma["MAE"])
            seed_att_cccs.append(ma["CCC"])

            seed_detail_rows.append({
                "SNR": int(s_val),
                "Model": f"Feature ELM (Seed {s})",
                "CBF_RMSE": mc["RMSE"], "CBF_MAE": mc["MAE"], "CBF_CCC": mc["CCC"],
                "ATT_RMSE": ma["RMSE"], "ATT_MAE": ma["MAE"], "ATT_CCC": ma["CCC"],
            })

        cbf_rmse_mean, cbf_rmse_std = np.mean(seed_cbf_rmses), np.std(seed_cbf_rmses)
        cbf_mae_mean, cbf_mae_std = np.mean(seed_cbf_maes), np.std(seed_cbf_maes)
        cbf_ccc_mean, cbf_ccc_std = np.mean(seed_cbf_cccs), np.std(seed_cbf_cccs)

        att_rmse_mean, att_rmse_std = np.mean(seed_att_rmses), np.std(seed_att_rmses)
        att_mae_mean, att_mae_std = np.mean(seed_att_maes), np.std(seed_att_maes)
        att_ccc_mean, att_ccc_std = np.mean(seed_att_cccs), np.std(seed_att_cccs)

        # Append DNN
        comparison_rows.append({
            "SNR": int(s_val),
            "Model": "DNN",
            "N_Samples": n_samples,
            "CBF_RMSE": m_dnn_c["RMSE"],
            "CBF_MAE": m_dnn_c["MAE"],
            "CBF_CCC": m_dnn_c["CCC"],
            "ATT_RMSE": m_dnn_a["RMSE"],
            "ATT_MAE": m_dnn_a["MAE"],
            "ATT_CCC": m_dnn_a["CCC"],
            "Variant_Note": "Baseline A (DNN-G, frozen)"
        })

        # Append Ensemble Feature ELM (Primary Defensible DNN-ELM)
        comparison_rows.append({
            "SNR": int(s_val),
            "Model": "DNN-ELM (Ensemble)",
            "N_Samples": n_samples,
            "CBF_RMSE": m_elm_ens_c["RMSE"],
            "CBF_MAE": m_elm_ens_c["MAE"],
            "CBF_CCC": m_elm_ens_c["CCC"],
            "ATT_RMSE": m_elm_ens_a["RMSE"],
            "ATT_MAE": m_elm_ens_a["MAE"],
            "ATT_CCC": m_elm_ens_a["CCC"],
            "Variant_Note": f"5-Seed Ensemble Feature ELM (Mean across seeds: CBF RMSE {cbf_rmse_mean:.4f}±{cbf_rmse_std:.4f}, ATT RMSE {att_rmse_mean:.4f}±{att_rmse_std:.4f})"
        })

        # Append Single-Seed Feature ELM (Seed 42)
        comparison_rows.append({
            "SNR": int(s_val),
            "Model": "DNN-ELM (Seed 42)",
            "N_Samples": n_samples,
            "CBF_RMSE": m_elm_s42_c["RMSE"],
            "CBF_MAE": m_elm_s42_c["MAE"],
            "CBF_CCC": m_elm_s42_c["CCC"],
            "ATT_RMSE": m_elm_s42_a["RMSE"],
            "ATT_MAE": m_elm_s42_a["MAE"],
            "ATT_CCC": m_elm_s42_a["CCC"],
            "Variant_Note": "Single-Seed Feature ELM (Seed 42 control)"
        })

    # Save to CSV
    df_comp = pd.DataFrame(comparison_rows)
    csv_path = os.path.join(OUT_DIR, "snr_10_15_20_metrics.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\nSaved SNR metrics to {csv_path}")

    df_seed_detail = pd.DataFrame(seed_detail_rows)
    csv_seed_path = os.path.join(OUT_DIR, "snr_seed_variability.csv")
    df_seed_detail.to_csv(csv_seed_path, index=False)
    print(f"Saved seed variability to {csv_seed_path}")

    # Generate Evaluation Report Markdown
    report_path = os.path.join(OUT_DIR, "EVALUATION_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Verified Evaluation Report: DNN vs DNN-ELM at SNR 10, 15, and 20\n\n")
        f.write("## 1. Verified Metrics Table\n\n")
        headers = ["SNR", "Model", "CBF_RMSE", "CBF_MAE", "CBF_CCC", "ATT_RMSE", "ATT_MAE", "ATT_CCC"]
        table_str = "| " + " | ".join(headers) + " |\n"
        table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df_comp.iterrows():
            row_vals = [
                str(row["SNR"]),
                str(row["Model"]),
                f"{row['CBF_RMSE']:.4f}",
                f"{row['CBF_MAE']:.4f}",
                f"{row['CBF_CCC']:.4f}",
                f"{row['ATT_RMSE']:.4f}",
                f"{row['ATT_MAE']:.4f}",
                f"{row['ATT_CCC']:.4f}"
            ]
            table_str += "| " + " | ".join(row_vals) + " |\n"
        f.write(table_str + "\n\n")

        f.write("## 2. Seed Variability (Mean ± SD across 5 seeds: 42, 43, 44, 45, 46)\n")
        f.write(f"- **SNR 10 dB:** CBF RMSE = `{cbf_rmse_mean:.4f} ± {cbf_rmse_std:.4f}`, ATT RMSE = `{att_rmse_mean:.4f} ± {att_rmse_std:.4f}`\n")
        f.write(f"- **SNR 15 dB:** CBF RMSE = `{cbf_rmse_mean:.4f} ± {cbf_rmse_std:.4f}`, ATT RMSE = `{att_rmse_mean:.4f} ± {att_rmse_std:.4f}`\n")
        f.write(f"- **SNR 20 dB:** CBF RMSE = `{cbf_rmse_mean:.4f} ± {cbf_rmse_std:.4f}`, ATT RMSE = `{att_rmse_mean:.4f} ± {att_rmse_std:.4f}`\n\n")

    print(f"Saved evaluation report to {report_path}")
    print("\nPhase 2-4 completed successfully.")

if __name__ == "__main__":
    run_evaluation()

