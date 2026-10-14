"""
run_ensemble_experiment.py
Phase 3: Controlled Ensemble-Size Experiment (K = 1, 3, 5, 10) at SNR 10, 15, and 20.
Features:
- Nested member sets: K=1 (member 1), K=3 (members 1-3), K=5 (members 1-5), K=10 (members 1-10).
- 5 independent pool repetitions (Seeds: 42..51, 142..151, 242..251, 342..351, 442..451).
- Evaluates CBF & ATT: RMSE, MAE, Bias, and Lin's CCC.
- Saves detailed run outputs to ensemble_size_metrics.csv.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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
    """Computes exact RMSE, MAE, Bias, and Lin's CCC."""
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
        "Bias": float(bias),
        "CCC": float(ccc)
    }

def run_experiment():
    print("=" * 80)
    print("PHASE 3: CONTROLLED ENSEMBLE-SIZE EXPERIMENT (K = 1, 3, 5, 10)")
    print("=" * 80)

    # 1. Load Pretrained DNN and Normalization
    norm_path = os.path.join(BASELINE_DIR, "normalization_stats.npz")
    cbf_ckpt_path = os.path.join(BASELINE_DIR, "CBFnet_G_baselineA.pth")
    att_ckpt_path = os.path.join(BASELINE_DIR, "ATTnet_G_baselineA.pth")

    norm_data = np.load(norm_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    cbf_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    att_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    cbf_net.load_state_dict(torch.load(cbf_ckpt_path, map_location=device, weights_only=False))
    att_net.load_state_dict(torch.load(att_ckpt_path, map_location=device, weights_only=False))
    cbf_net.eval(); att_net.eval()

    # 2. Load Datasets
    print("Loading Dataset arrays...")
    N_tr = 100_000
    X_tr_raw = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')[:N_tr]
    Y_tr_raw = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')[:N_tr]
    X_test_raw = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')[:]
    Y_test_raw = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')[:]
    snr_test = np.load(os.path.join(CACHE_DIR, "snr_test.npy"), mmap_mode='r')[:]

    X_tr_n = ((X_tr_raw - X_mean) / X_std).astype(np.float32)
    X_test_n = ((X_test_raw - X_mean) / X_std).astype(np.float32)

    # 3. Extract Penultimate Features (Reused across all members and repeats)
    print("\nExtracting Penultimate Features from frozen DNN...")
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

    # Baseline DNN predictions on test set
    pred_cbf_dnn = cbf_net.predict_physical(X_test_n)
    pred_att_dnn = att_net.predict_physical(X_test_n)

    # 4. Multi-Repeat Nested Ensemble Study
    # 5 independent repetitions of 10-member pools
    repeats = 5
    ensemble_sizes = [1, 3, 5, 10]
    snr_targets = [10.0, 15.0, 20.0]

    all_results = []
    t0_study = time.time()

    for r in range(1, repeats + 1):
        print(f"\n--- Repeat {r}/{repeats}: Training 10-member ELM Pool ---")
        base_seed = 42 + (r - 1) * 100
        pool_seeds = [base_seed + i for i in range(10)]
        print(f"  Seeds in Pool {r}: {pool_seeds}")

        cbf_pool_preds = []
        att_pool_preds = []

        for s in pool_seeds:
            # Train CBF ELM
            elm_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, activation="tanh", seed=s)
            elm_c.fit_chunked(F_cbf_tr, Y_cbf_tr_std, chunk_size=50_000)
            p_c = elm_c.predict(F_cbf_te, chunk_size=50_000)
            cbf_pool_preds.append(np.clip(p_c * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX))

            # Train ATT ELM
            elm_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, activation="tanh", seed=s)
            elm_a.fit_chunked(F_att_tr, Y_att_tr_std, chunk_size=50_000)
            p_a = elm_a.predict(F_att_te, chunk_size=50_000)
            att_pool_preds.append(np.clip(p_a * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX))
            del elm_c, elm_a

        # Evaluate nested subsets: K=1 (pool[0]), K=3 (pool[0..3]), K=5 (pool[0..5]), K=10 (pool[0..10])
        for K in ensemble_sizes:
            cbf_ens_k = np.mean(cbf_pool_preds[:K], axis=0)
            att_ens_k = np.mean(att_pool_preds[:K], axis=0)

            for s_val in snr_targets:
                mask = (snr_test == s_val)
                n_samples = int(np.sum(mask))

                y_true_cbf = Y_test_raw[mask, 0]
                y_true_att = Y_test_raw[mask, 1]

                # Compute ELM ensemble metrics
                mc_elm = calculate_exact_metrics(y_true_cbf, cbf_ens_k[mask])
                ma_elm = calculate_exact_metrics(y_true_att, att_ens_k[mask])

                # Compute baseline DNN metrics
                mc_dnn = calculate_exact_metrics(y_true_cbf, pred_cbf_dnn[mask])
                ma_dnn = calculate_exact_metrics(y_true_att, pred_att_dnn[mask])

                all_results.append({
                    "Repeat": r,
                    "Ensemble_Size_K": K,
                    "SNR": int(s_val),
                    "N_Samples": n_samples,
                    "CBF_RMSE": mc_elm["RMSE"],
                    "CBF_MAE": mc_elm["MAE"],
                    "CBF_Bias": mc_elm["Bias"],
                    "CBF_CCC": mc_elm["CCC"],
                    "ATT_RMSE": ma_elm["RMSE"],
                    "ATT_MAE": ma_elm["MAE"],
                    "ATT_Bias": ma_elm["Bias"],
                    "ATT_CCC": ma_elm["CCC"],
                    "DNN_CBF_RMSE": mc_dnn["RMSE"],
                    "DNN_CBF_MAE": mc_dnn["MAE"],
                    "DNN_CBF_CCC": mc_dnn["CCC"],
                    "DNN_ATT_RMSE": ma_dnn["RMSE"],
                    "DNN_ATT_MAE": ma_dnn["MAE"],
                    "DNN_ATT_CCC": ma_dnn["CCC"],
                    "Member_Seeds": str(pool_seeds[:K])
                })

    print(f"\nAll {repeats} pool repetitions completed in {time.time() - t0_study:.2f} seconds.")

    # Save detailed runs to CSV
    df_runs = pd.DataFrame(all_results)
    csv_runs_path = os.path.join(OUT_DIR, "ensemble_size_metrics.csv")
    df_runs.to_csv(csv_runs_path, index=False)
    print(f"Saved detailed metrics to {csv_runs_path}")

    # Also save primary pool (Repeat 1) predictions for diversity analysis
    # Save repeat 1 member predictions (N=100k, 10 members)
    pred_save_path = os.path.join(OUT_DIR, "repeat1_member_predictions.npz")
    # For storage efficiency, only save the test indices corresponding to SNR 10, 15, 20
    test_snr_mask = np.isin(snr_test, [10.0, 15.0, 20.0])
    # Compute member predictions for repeat 1
    # We will re-save in diversity script or save here
    print("\nPhase 3 experiment completed successfully.")

if __name__ == "__main__":
    run_experiment()

