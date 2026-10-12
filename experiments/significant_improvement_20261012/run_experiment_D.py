"""
run_experiment_D.py
Experiment D: DNN-Feature plus Physics-Feature Residual ELM.
Features:
  - DNN penultimate latents (128-dim CBF, 64-dim ATT)
  - 8 ASL domain physics features (inter-PLD slopes, decay ratios, AUC, early/late ratio)
Out-of-Fold (OOF) residual training to eliminate in-sample leakage:
  - 5-fold cross-validation to generate true OOF DNN predictions and residuals.
Models compared:
  1. Baseline A DNN-G
  2. Existing Ensemble Feature ELM (Latents only, 5 seeds)
  3. Physics-Feature ELM (8 Physics features only)
  4. Combined Latent + Physics Feature ELM
  5. Out-of-Fold Residual Physics-Feature ELM
Strict validation evaluation on locked Validation Set (N=100,000). Test set locked.
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

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
ELM_DIR = os.path.join(EXP_DIR, "elm_models")
OUT_DIR = os.path.join(EXP_DIR, "significant_improvement_20261012")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

sys.path.append(EXP_DIR)
sys.path.append(ELM_DIR)
import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

def extract_physics_features(X_raw):
    """
    Extract 8 ASL physics-justified features from 4-PLD signals:
      1-3: Inter-PLD temporal slopes: (x2 - x1), (x3 - x2), (x4 - x3)
      4-6: Successive decay ratios: x2/(x1+eps), x3/(x2+eps), x4/(x3+eps)
      7:   Normalized signal AUC: sum(x) / mean(x)
      8:   Early-to-late perfusion ratio: (x1 + x2) / (x3 + x4 + eps)
    """
    eps = 1e-3
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]
    x3 = X_raw[:, 2]
    x4 = X_raw[:, 3]

    d21 = x2 - x1
    d32 = x3 - x2
    d43 = x4 - x3

    r21 = x2 / (x1 + eps)
    r32 = x3 / (x2 + eps)
    r43 = x4 / (x3 + eps)

    total = x1 + x2 + x3 + x4
    mean_val = np.maximum(np.mean(X_raw, axis=1), eps)
    auc_norm = total / (mean_val * 4.0)

    early_late = (x1 + x2) / (x3 + x4 + eps)

    feats = np.stack([d21, d32, d43, r21, r32, r43, auc_norm, early_late], axis=1)
    return feats.astype(np.float32)

def main():
    print("=" * 80)
    print("EXPERIMENT D: DNN-FEATURE + PHYSICS-FEATURE RESIDUAL ELM (WITH OOF)")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Normalization parameters
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    # 2. Load Baseline A Networks
    att_net = StandardizedNet(len(physics.PLDS), 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device))
    att_net.eval()

    cbf_net = StandardizedNet(len(physics.PLDS), 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device))
    cbf_net.eval()

    # 3. Load Datasets
    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    # Training slice: 100k for OOF and ELM training
    N_tr = 100_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    X_tr_raw = x_tr_mmap[idx_tr]
    Y_cbf_tr = y_tr_mmap[idx_tr, 0]
    Y_att_tr = y_tr_mmap[idx_tr, 1]
    Y_cbf_tr_std = ((Y_cbf_tr - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_tr_std = ((Y_att_tr - att_mean) / att_std).astype(np.float32)

    X_val_raw = x_val_mmap[:]
    X_val_n = ((X_val_raw - X_mean) / X_std).astype(np.float32)
    Y_cbf_val = y_val_mmap[:, 0]
    Y_att_val = y_val_mmap[:, 1]

    # Baseline A validation predictions (Control)
    p_cbf_base_val = cbf_net.predict_physical(X_val_n)
    p_att_base_val = att_net.predict_physical(X_val_n)

    m_base_c = calc_metrics(Y_cbf_val, p_cbf_base_val)
    m_base_a = calc_metrics(Y_att_val, p_att_base_val)
    print(f"Control: Baseline A DNN-G Val:")
    print(f"  CBF Val RMSE: {m_base_c['RMSE']:.4f}, MAE: {m_base_c['MAE']:.4f}, CCC: {m_base_c['CCC']:.4f}")
    print(f"  ATT Val RMSE: {m_base_a['RMSE']:.4f} s, MAE: {m_base_a['MAE']:.4f} s, CCC: {m_base_a['CCC']:.4f}")

    # 4. Extract Features
    print("\nExtracting Physics Features (8 dimensions)...")
    F_phys_tr = extract_physics_features(X_tr_raw)
    F_phys_val = extract_physics_features(X_val_raw)

    # Standardize physics features on training set
    mu_phys = np.mean(F_phys_tr, axis=0, keepdims=True)
    std_phys = np.std(F_phys_tr, axis=0, keepdims=True) + 1e-6
    F_phys_tr_n = (F_phys_tr - mu_phys) / std_phys
    F_phys_val_n = (F_phys_val - mu_phys) / std_phys

    print("Extracting DNN Penultimate Latents...")
    X_tr_n = ((X_tr_raw - X_mean) / X_std).astype(np.float32)

    with torch.no_grad():
        F_lat_cbf_tr = cbf_net.penultimate_features(torch.from_numpy(X_tr_n).to(device))
        F_lat_cbf_val = cbf_net.penultimate_features(torch.from_numpy(X_val_n).to(device))

        F_lat_att_tr = att_net.penultimate_features(torch.from_numpy(X_tr_n).to(device))
        F_lat_att_val = att_net.penultimate_features(torch.from_numpy(X_val_n).to(device))

    # Model 1: Existing Latent-Only ELM (Ensemble 5 seeds)
    print("\nTraining Model 1: Existing Latent-Only Feature ELM (5 seeds)...")
    p_cbf_lat_ens = np.zeros(len(Y_cbf_val), dtype=np.float32)
    p_att_lat_ens = np.zeros(len(Y_att_val), dtype=np.float32)

    for s in [42, 43, 44, 45, 46]:
        e_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_c.fit_chunked(F_lat_cbf_tr, Y_cbf_tr_std)
        p_cbf_lat_ens += np.clip(e_c.predict(F_lat_cbf_val) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)

        e_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_a.fit_chunked(F_lat_att_tr, Y_att_tr_std)
        p_att_lat_ens += np.clip(e_a.predict(F_lat_att_val) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

    p_cbf_lat_ens /= 5.0
    p_att_lat_ens /= 5.0

    m_lat_c = calc_metrics(Y_cbf_val, p_cbf_lat_ens)
    m_lat_a = calc_metrics(Y_att_val, p_att_lat_ens)
    print(f"  Model 1 (Latent ELM) -> CBF RMSE: {m_lat_c['RMSE']:.4f}, MAE: {m_lat_c['MAE']:.4f} | ATT RMSE: {m_lat_a['RMSE']:.4f} s, MAE: {m_lat_a['MAE']:.4f} s")

    # Model 2: Physics-Features Only ELM
    print("\nTraining Model 2: Physics-Features Only ELM...")
    p_cbf_phys_ens = np.zeros(len(Y_cbf_val), dtype=np.float32)
    p_att_phys_ens = np.zeros(len(Y_att_val), dtype=np.float32)

    for s in [42, 43, 44, 45, 46]:
        e_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_c.fit_chunked(F_phys_tr_n, Y_cbf_tr_std)
        p_cbf_phys_ens += np.clip(e_c.predict(F_phys_val_n) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)

        e_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_a.fit_chunked(F_phys_tr_n, Y_att_tr_std)
        p_att_phys_ens += np.clip(e_a.predict(F_phys_val_n) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

    p_cbf_phys_ens /= 5.0
    p_att_phys_ens /= 5.0

    m_phys_c = calc_metrics(Y_cbf_val, p_cbf_phys_ens)
    m_phys_a = calc_metrics(Y_att_val, p_att_phys_ens)
    print(f"  Model 2 (Physics-Only ELM) -> CBF RMSE: {m_phys_c['RMSE']:.4f}, MAE: {m_phys_c['MAE']:.4f} | ATT RMSE: {m_phys_a['RMSE']:.4f} s, MAE: {m_phys_a['MAE']:.4f} s")

    # Model 3: Combined Latent + Physics Feature ELM
    print("\nTraining Model 3: Combined Latent + Physics Feature ELM...")
    F_comb_cbf_tr = np.hstack([F_lat_cbf_tr, F_phys_tr_n])
    F_comb_cbf_val = np.hstack([F_lat_cbf_val, F_phys_val_n])

    F_comb_att_tr = np.hstack([F_lat_att_tr, F_phys_tr_n])
    F_comb_att_val = np.hstack([F_lat_att_val, F_phys_val_n])

    p_cbf_comb_ens = np.zeros(len(Y_cbf_val), dtype=np.float32)
    p_att_comb_ens = np.zeros(len(Y_att_val), dtype=np.float32)

    for s in [42, 43, 44, 45, 46]:
        e_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_c.fit_chunked(F_comb_cbf_tr, Y_cbf_tr_std)
        p_cbf_comb_ens += np.clip(e_c.predict(F_comb_cbf_val) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)

        e_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_a.fit_chunked(F_comb_att_tr, Y_att_tr_std)
        p_att_comb_ens += np.clip(e_a.predict(F_comb_att_val) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

    p_cbf_comb_ens /= 5.0
    p_att_comb_ens /= 5.0

    m_comb_c = calc_metrics(Y_cbf_val, p_cbf_comb_ens)
    m_comb_a = calc_metrics(Y_att_val, p_att_comb_ens)
    print(f"  Model 3 (Combined Latent+Physics ELM) -> CBF RMSE: {m_comb_c['RMSE']:.4f}, MAE: {m_comb_c['MAE']:.4f} | ATT RMSE: {m_comb_a['RMSE']:.4f} s, MAE: {m_comb_a['MAE']:.4f} s")

    # Model 4: Out-of-Fold (OOF) Residual Physics-Feature ELM
    print("\nTraining Model 4: Out-of-Fold (OOF) Residual ELM...")
    # Generate 5-fold cross-validated predictions on the 100k training set
    K = 5
    fold_size = N_tr // K
    perm_oof = rng.permutation(N_tr)
    oof_cbf_pred = np.empty(N_tr, dtype=np.float32)
    oof_att_pred = np.empty(N_tr, dtype=np.float32)

    # Fast 5-fold DNN CV on the 100k set to get true OOF predictions
    criterion_mse = nn.MSELoss()
    for k in range(K):
        val_mask = np.zeros(N_tr, dtype=bool)
        val_mask[perm_oof[k*fold_size : (k+1)*fold_size]] = True
        tr_mask = ~val_mask

        # Train fold CBFnet
        net_c_fold = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
        opt_c = torch.optim.Adam(net_c_fold.parameters(), lr=1e-3)
        # Train fold ATTnet
        net_a_fold = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
        opt_a = torch.optim.Adam(net_a_fold.parameters(), lr=1e-3)

        x_fold_tr_t = torch.from_numpy(X_tr_n[tr_mask]).to(device)
        y_cbf_fold_tr_t = torch.from_numpy(Y_cbf_tr_std[tr_mask]).unsqueeze(1).to(device)
        y_att_fold_tr_t = torch.from_numpy(Y_att_tr_std[tr_mask]).unsqueeze(1).to(device)

        # Train 12 fast epochs per fold
        b_sz = 1024
        n_b = int(np.ceil(np.sum(tr_mask) / b_sz))
        for ep in range(12):
            for b in range(n_b):
                bx = x_fold_tr_t[b*b_sz:(b+1)*b_sz]
                opt_c.zero_grad()
                loss_c = criterion_mse(net_c_fold(bx), y_cbf_fold_tr_t[b*b_sz:(b+1)*b_sz])
                loss_c.backward()
                opt_c.step()

                opt_a.zero_grad()
                loss_a = criterion_mse(net_a_fold(bx), y_att_fold_tr_t[b*b_sz:(b+1)*b_sz])
                loss_a.backward()
                opt_a.step()

        net_c_fold.eval()
        net_a_fold.eval()
        with torch.no_grad():
            oof_cbf_pred[val_mask] = net_c_fold.predict_physical(X_tr_n[val_mask])
            oof_att_pred[val_mask] = net_a_fold.predict_physical(X_tr_n[val_mask])

    # OOF Residuals (Free of in-sample memorization!)
    oof_cbf_residual = Y_cbf_tr - oof_cbf_pred
    oof_att_residual = Y_att_tr - oof_att_pred

    # Train Residual ELM on OOF residuals using physics features + latents
    e_res_cbf = GPUChunkedELM(n_hidden=512, alpha=1.0, seed=42)
    e_res_cbf.fit_chunked(F_comb_cbf_tr, oof_cbf_residual)

    e_res_att = GPUChunkedELM(n_hidden=512, alpha=1.0, seed=42)
    e_res_att.fit_chunked(F_comb_att_tr, oof_att_residual)

    # Predict residual on validation set
    pred_res_cbf_val = e_res_cbf.predict(F_comb_cbf_val)
    pred_res_att_val = e_res_att.predict(F_comb_att_val)

    # Validation predictions with OOF residual correction
    p_cbf_oof_res_val = np.clip(p_cbf_base_val + 0.5 * pred_res_cbf_val, physics.CBF_MIN, physics.CBF_MAX)
    p_att_oof_res_val = np.clip(p_att_base_val + 0.5 * pred_res_att_val, physics.ATT_MIN, physics.ATT_MAX)

    m_oof_c = calc_metrics(Y_cbf_val, p_cbf_oof_res_val)
    m_oof_a = calc_metrics(Y_att_val, p_att_oof_res_val)
    print(f"  Model 4 (OOF Residual ELM) -> CBF RMSE: {m_oof_c['RMSE']:.4f}, MAE: {m_oof_c['MAE']:.4f} | ATT RMSE: {m_oof_a['RMSE']:.4f} s, MAE: {m_oof_a['MAE']:.4f} s")

    # Summary Table
    exp_d_rows = [
        {"Model": "Baseline A (DNN-G)", "CBF_RMSE": m_base_c["RMSE"], "CBF_MAE": m_base_c["MAE"], "CBF_CCC": m_base_c["CCC"],
         "ATT_RMSE": m_base_a["RMSE"], "ATT_MAE": m_base_a["MAE"], "ATT_CCC": m_base_a["CCC"]},
        {"Model": "Existing Latent ELM (5 seeds)", "CBF_RMSE": m_lat_c["RMSE"], "CBF_MAE": m_lat_c["MAE"], "CBF_CCC": m_lat_c["CCC"],
         "ATT_RMSE": m_lat_a["RMSE"], "ATT_MAE": m_lat_a["MAE"], "ATT_CCC": m_lat_a["CCC"]},
        {"Model": "Physics-Features Only ELM", "CBF_RMSE": m_phys_c["RMSE"], "CBF_MAE": m_phys_c["MAE"], "CBF_CCC": m_phys_c["CCC"],
         "ATT_RMSE": m_phys_a["RMSE"], "ATT_MAE": m_phys_a["MAE"], "ATT_CCC": m_phys_a["CCC"]},
        {"Model": "Combined Latent + Physics ELM", "CBF_RMSE": m_comb_c["RMSE"], "CBF_MAE": m_comb_c["MAE"], "CBF_CCC": m_comb_c["CCC"],
         "ATT_RMSE": m_comb_a["RMSE"], "ATT_MAE": m_comb_a["MAE"], "ATT_CCC": m_comb_a["CCC"]},
        {"Model": "OOF Residual Physics ELM", "CBF_RMSE": m_oof_c["RMSE"], "CBF_MAE": m_oof_c["MAE"], "CBF_CCC": m_oof_c["CCC"],
         "ATT_RMSE": m_oof_a["RMSE"], "ATT_MAE": m_oof_a["MAE"], "ATT_CCC": m_oof_a["CCC"]},
    ]
    df_d = pd.DataFrame(exp_d_rows)
    csv_out = os.path.join(TABLES_DIR, "experiment_D_val_results.csv")
    df_d.to_csv(csv_out, index=False)
    print(f"\nSaved Experiment D Validation Results to: {csv_out}")
    print(df_d.to_string(index=False))
    print("=" * 80)
    print("EXPERIMENT D COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
