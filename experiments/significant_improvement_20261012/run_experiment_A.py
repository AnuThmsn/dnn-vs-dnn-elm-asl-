"""
run_experiment_A.py
Experiment A: SNR-Balanced and Difficulty-Aware Training Sampling.
Evaluates:
  1. Standard uniform sampling (Control)
  2. SNR-balanced sampling (stratified across 12 SNR levels)
  3. SNR-balanced + ATT difficulty-aware sampling (oversampling early-arrival [0.5, 1.2] s)
Evaluates strictly on Validation Set (N=100,000). Keeps test set locked.
Storage strictly on Drive D; host RAM < 80 MB.
"""

import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
OUT_DIR = os.path.join(EXP_DIR, "significant_improvement_20261012")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

sys.path.append(EXP_DIR)
import physics
from train_baseline_A import StandardizedNet, calc_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 80)
    print("EXPERIMENT A: SNR-BALANCED & DIFFICULTY-AWARE SAMPLING")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Normalization Stats
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    # 2. Memory-map Training and Validation Arrays
    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    snr_tr_mmap = np.load(os.path.join(CACHE_DIR, "snr_train.npy"), mmap_mode='r')

    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    # Normalized Validation Data
    X_val_n = ((x_val_mmap[:] - X_mean) / X_std).astype(np.float32)
    Y_cbf_val = y_val_mmap[:, 0]
    Y_att_val = y_val_mmap[:, 1]
    Y_cbf_val_std = ((Y_cbf_val - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_val_std = ((Y_att_val - att_mean) / att_std).astype(np.float32)

    # Training size: use representative 250,000-sample balanced slices for fast, fair comparison
    N_tr_sub = 250_000
    rng = np.random.default_rng(2026)

    # Sampling Strategy 1: Standard Uniform Random Selection
    idx_uniform = rng.choice(1_000_000, size=N_tr_sub, replace=False)

    # Sampling Strategy 2: SNR-Balanced Selection (equal samples across 12 SNR levels)
    snr_levels = physics.SNR_VALUES
    idx_snr_balanced = []
    per_snr = N_tr_sub // len(snr_levels)
    for s in snr_levels:
        mask_s = np.isinf(snr_tr_mmap) if np.isinf(s) else (snr_tr_mmap == s)
        avail = np.where(mask_s)[0]
        chosen = rng.choice(avail, size=per_snr, replace=False)
        idx_snr_balanced.append(chosen)
    idx_snr_balanced = np.concatenate(idx_snr_balanced)

    # Sampling Strategy 3: SNR-Balanced + Difficulty-Aware (2x weight for ATT in [0.5, 1.2] s)
    idx_diff_aware = []
    for s in snr_levels:
        mask_s = np.isinf(snr_tr_mmap) if np.isinf(s) else (snr_tr_mmap == s)
        avail = np.where(mask_s)[0]
        att_avail = y_tr_mmap[avail, 1]
        weights = np.where(att_avail <= 1.2, 2.0, 1.0)
        probs = weights / weights.sum()
        chosen = rng.choice(avail, size=per_snr, replace=False, p=probs)
        idx_diff_aware.append(chosen)
    idx_diff_aware = np.concatenate(idx_diff_aware)

    strategies = [
        ("A1_Uniform_Control", idx_uniform),
        ("A2_SNR_Balanced", idx_snr_balanced),
        ("A3_Difficulty_Aware", idx_diff_aware)
    ]

    val_records = []
    criterion = nn.MSELoss()

    for strat_name, idx_train in strategies:
        print(f"\n--- Training Strategy: {strat_name} (N={len(idx_train):,}) ---")
        X_sub_n = torch.from_numpy(((x_tr_mmap[idx_train] - X_mean) / X_std).astype(np.float32)).to(device)
        Y_cbf_sub_std = torch.from_numpy(((y_tr_mmap[idx_train, 0] - cbf_mean) / cbf_std).astype(np.float32)).unsqueeze(1).to(device)
        Y_att_sub_std = torch.from_numpy(((y_tr_mmap[idx_train, 1] - att_mean) / att_std).astype(np.float32)).unsqueeze(1).to(device)

        X_val_t = torch.from_numpy(X_val_n).to(device)
        Y_cbf_val_t = torch.from_numpy(Y_cbf_val_std).unsqueeze(1).to(device)
        Y_att_val_t = torch.from_numpy(Y_att_val_std).unsqueeze(1).to(device)

        # Train ATTnet (4x64)
        att_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
        opt_att = torch.optim.Adam(att_net.parameters(), lr=1e-3)
        best_att_val = float("inf")
        best_att_state = None

        batch_size = 1024
        n_batches = int(np.ceil(len(idx_train) / batch_size))
        t0 = time.time()

        for ep in range(1, 26):
            att_net.train()
            perm = torch.randperm(len(idx_train), device=device)
            for b in range(n_batches):
                b_idx = perm[b*batch_size : (b+1)*batch_size]
                opt_att.zero_grad()
                pred = att_net(X_sub_n[b_idx])
                loss = criterion(pred, Y_att_sub_std[b_idx])
                loss.backward()
                opt_att.step()

            att_net.eval()
            with torch.no_grad():
                val_loss = criterion(att_net(X_val_t), Y_att_val_t).item()
            if val_loss < best_att_val:
                best_att_val = val_loss
                best_att_state = {k: v.cpu().clone() for k, v in att_net.state_dict().items()}

        att_train_time = time.time() - t0
        att_net.load_state_dict(best_att_state)
        att_net.eval()
        p_att_val = att_net.predict_physical(X_val_n)
        m_att = calc_metrics(Y_att_val, p_att_val)
        print(f"  ATT ({strat_name}) -> Val RMSE: {m_att['RMSE']:.4f} s, MAE: {m_att['MAE']:.4f} s, CCC: {m_att['CCC']:.4f} (Time: {att_train_time:.1f}s)")

        # Train CBFnet (5x128)
        cbf_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
        opt_cbf = torch.optim.Adam(cbf_net.parameters(), lr=1e-3)
        best_cbf_val = float("inf")
        best_cbf_state = None
        t0 = time.time()

        for ep in range(1, 26):
            cbf_net.train()
            perm = torch.randperm(len(idx_train), device=device)
            for b in range(n_batches):
                b_idx = perm[b*batch_size : (b+1)*batch_size]
                opt_cbf.zero_grad()
                pred = cbf_net(X_sub_n[b_idx])
                loss = criterion(pred, Y_cbf_sub_std[b_idx])
                loss.backward()
                opt_cbf.step()

            cbf_net.eval()
            with torch.no_grad():
                val_loss = criterion(cbf_net(X_val_t), Y_cbf_val_t).item()
            if val_loss < best_cbf_val:
                best_cbf_val = val_loss
                best_cbf_state = {k: v.cpu().clone() for k, v in cbf_net.state_dict().items()}

        cbf_train_time = time.time() - t0
        cbf_net.load_state_dict(best_cbf_state)
        cbf_net.eval()
        p_cbf_val = cbf_net.predict_physical(X_val_n)
        m_cbf = calc_metrics(Y_cbf_val, p_cbf_val)
        print(f"  CBF ({strat_name}) -> Val RMSE: {m_cbf['RMSE']:.4f}, MAE: {m_cbf['MAE']:.4f}, CCC: {m_cbf['CCC']:.4f} (Time: {cbf_train_time:.1f}s)")

        # Save Checkpoint
        torch.save(best_att_state, os.path.join(CKPT_DIR, f"expA_att_{strat_name}.pth"))
        torch.save(best_cbf_state, os.path.join(CKPT_DIR, f"expA_cbf_{strat_name}.pth"))

        val_records.append({
            "Experiment": "Experiment A (Sampling)",
            "Configuration": strat_name,
            "CBF_Val_RMSE": m_cbf["RMSE"],
            "CBF_Val_MAE": m_cbf["MAE"],
            "CBF_Val_CCC": m_cbf["CCC"],
            "CBF_Val_R2": m_cbf["R2"],
            "ATT_Val_RMSE": m_att["RMSE"],
            "ATT_Val_MAE": m_att["MAE"],
            "ATT_Val_CCC": m_att["CCC"],
            "ATT_Val_R2": m_att["R2"],
            "Train_Time_Sec": att_train_time + cbf_train_time
        })

        del X_sub_n, Y_cbf_sub_std, Y_att_sub_std
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_res = pd.DataFrame(val_records)
    csv_out = os.path.join(TABLES_DIR, "experiment_A_val_results.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved Experiment A Validation Results to: {csv_out}")
    print(df_res.to_string(index=False))
    print("=" * 80)
    print("EXPERIMENT A COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

