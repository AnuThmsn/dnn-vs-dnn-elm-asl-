"""
train_optimized_acquisition.py
Phase 5: Train Candidate Estimators on Optimized 4-PLD Acquisition [0.40, 1.10, 2.00, 3.00] s.
- Generates matched multi-PLD signals on Drive D using identical ground-truth (Y_train, Y_val, Y_test)
  and matched noise seeds (seed 42 for train, 99 for val, 7 for test).
- Computes training-only normalization statistics (X_mean, X_std).
- Trains exact matched architectures:
    * ATTnet_opt: 4 inputs, 4 hidden layers x 64 neurons (ELU, MSE loss)
    * CBFnet_opt: 4 inputs, 5 hidden layers x 128 neurons (ELU, MSE loss)
- Fits 5-seed Ensemble Feature ELM readout on candidate DNN penultimate activations.
- Strict Drive D memory safeguards (< 90 MB host RAM).
"""

import os
import sys
import gc
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
ELM_DIR = os.path.join(EXP_DIR, "elm_models")
OUT_DIR = os.path.join(EXP_DIR, "acquisition_optimization_20261013")
OPT_CACHE_DIR = os.path.join(OUT_DIR, "cache")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
TABLES_DIR = os.path.join(OUT_DIR, "tables")

for d in [OPT_CACHE_DIR, CKPT_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.append(EXP_DIR)
sys.path.append(ELM_DIR)
sys.path.append(OUT_DIR)
import physics
import memory_config
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

OPTIMIZED_PLDS = np.array([0.40, 1.10, 2.00, 3.00], dtype=np.float64)

def generate_optimized_signals_split(split_name, seed, n_samples, chunk_size=200_000):
    """
    Generates signals for OPTIMIZED_PLDS using the exact underlying Y and SNR from CACHE_DIR.
    Stores array directly on Drive D using memory mapping.
    """
    x_opt_path = os.path.join(OPT_CACHE_DIR, f"X_{split_name}_opt.npy")
    if os.path.exists(x_opt_path) and os.path.getsize(x_opt_path) == n_samples * 4 * 4: # float32
        print(f"Loading existing cache for {split_name}_opt from {x_opt_path}...")
        return x_opt_path

    y_orig = np.load(os.path.join(CACHE_DIR, f"Y_{split_name}.npy"), mmap_mode='r')
    snr_orig = np.load(os.path.join(CACHE_DIR, f"snr_{split_name}.npy"), mmap_mode='r')

    print(f"Generating optimized signals for {split_name} (N={n_samples:,}, seed={seed})...")
    rng = np.random.default_rng(seed)

    x_mmap = np.lib.format.open_memmap(x_opt_path, mode='w+', dtype=np.float32, shape=(n_samples, 4))

    n_chunks = int(np.ceil(n_samples / chunk_size))
    for c_idx in range(n_chunks):
        start = c_idx * chunk_size
        end = min(start + chunk_size, n_samples)
        c_len = end - start

        cbf_c = y_orig[start:end, 0].astype(np.float64)
        att_c = y_orig[start:end, 1].astype(np.float64)
        snr_c = snr_orig[start:end].astype(np.float64)

        clean_sig_c = physics.compute_physical_signals(OPTIMIZED_PLDS, physics.TAU, cbf_c, att_c)
        X_c = physics.add_rician_noise(clean_sig_c, snr_c, rng)

        x_mmap[start:end] = X_c
        x_mmap.flush()

        del cbf_c, att_c, snr_c, clean_sig_c, X_c
        gc.collect()

    del x_mmap
    gc.collect()
    print(f"Finished caching {x_opt_path}.")
    return x_opt_path

def train_standardized_model(net, X_tr_n, Y_col_tr, X_val_n, Y_col_val,
                             batch_size=512, max_epochs=200, patience=10, lr=1e-3, name="Model"):
    Y_tr_std = ((Y_col_tr - net.y_mean) / net.y_std).astype(np.float32)
    Y_val_std = ((Y_col_val - net.y_mean) / net.y_std).astype(np.float32)

    ds_tr = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr_std).unsqueeze(1))
    ds_val = TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(Y_val_std).unsqueeze(1))

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, pin_memory=False)
    ld_val = DataLoader(ds_val, batch_size=batch_size, pin_memory=False)

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    no_imp = 0
    history = []

    t0 = time.time()
    for epoch in range(max_epochs):
        net.train()
        tr = 0.0
        for xb, yb in ld_tr:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optim.zero_grad()
            pred = net(xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            tr += loss.item()
        tr /= len(ld_tr)

        net.eval()
        with torch.no_grad():
            vl = 0.0
            for xb, yb in ld_val:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                vl += crit(net(xb), yb).item()
            vl /= len(ld_val)

        history.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": vl})
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(f"  [{name}] Epoch {epoch+1:3d}: train={tr:.6f} val={vl:.6f}")

        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(net.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  [{name}] Early stopping at epoch {epoch+1} (patience={patience})")
                break

    train_time = time.time() - t0
    print(f"[{name}] Training completed in {train_time:.1f}s ({len(history)} epochs, best_val={best_val:.6f})")
    net.load_state_dict(best_state)

    del ds_tr, ds_val, ld_tr, ld_val, optim, crit, Y_tr_std, Y_val_std
    memory_config.cleanup_memory()
    return net, history, train_time

def main():
    print("=" * 80)
    print("PHASE 5: TRAINING ESTIMATORS ON OPTIMIZED 4-PLD ACQUISITION")
    print(f"Target PLDs: {OPTIMIZED_PLDS.tolist()}")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Generate & Cache Optimized Signals
    x_tr_path = generate_optimized_signals_split("train", seed=42, n_samples=1_000_000)
    x_val_path = generate_optimized_signals_split("val", seed=99, n_samples=100_000)
    x_test_path = generate_optimized_signals_split("test", seed=7, n_samples=100_000)

    # 2. Compute Training-Only Normalization Statistics
    print("\nComputing training-only normalization statistics on X_train_opt...")
    x_tr_mmap = np.load(x_tr_path, mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(x_val_path, mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    sum_x = np.zeros(4, dtype=np.float64)
    sum_sq_x = np.zeros(4, dtype=np.float64)
    chunk = 100_000
    for i in range(0, 1_000_000, chunk):
        xc = x_tr_mmap[i:i+chunk]
        sum_x += xc.sum(axis=0)
        sum_sq_x += (xc.astype(np.float64)**2).sum(axis=0)

    X_mean_opt = (sum_x / 1_000_000).astype(np.float32).reshape(1, 4)
    X_std_opt = np.sqrt((sum_sq_x / 1_000_000) - (X_mean_opt.squeeze()**2)).astype(np.float32).reshape(1, 4)

    # Load target normalization from existing cache
    norm_base = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    cbf_mean = float(norm_base["cbf_mean"])
    cbf_std = float(norm_base["cbf_std"])
    att_mean = float(norm_base["att_mean"])
    att_std = float(norm_base["att_std"])

    norm_save_path = os.path.join(CKPT_DIR, "normalization_stats_opt.npz")
    np.savez(norm_save_path,
             X_mean=X_mean_opt, X_std=X_std_opt,
             cbf_mean=cbf_mean, cbf_std=cbf_std,
             att_mean=att_mean, att_std=att_std)
    print(f"Saved optimized normalization statistics to: {norm_save_path}")

    # 3. Prepare normalized arrays for memory-efficient training
    print("\nPreparing normalized training inputs in RAM (16 MB)...")
    X_tr_n = ((x_tr_mmap[:] - X_mean_opt) / X_std_opt).astype(np.float32)
    X_val_n = ((x_val_mmap[:] - X_mean_opt) / X_std_opt).astype(np.float32)

    # 4. Train ATTnet_opt (4 hidden layers x 64 neurons, MSE loss)
    print("\n" + "=" * 60)
    print("Training ATTnet_opt (4x64, MSE loss, Optimized PLDs)...")
    print("=" * 60)
    att_opt_net = StandardizedNet(4, n_hidden_layers=4, n_neurons=64,
                                  y_mean=att_mean, y_std=att_std,
                                  y_min=physics.ATT_MIN, y_max=physics.ATT_MAX).to(device)

    Y_att_tr = y_tr_mmap[:, 1]
    Y_att_val = y_val_mmap[:, 1]
    att_opt_net, att_hist, att_time = train_standardized_model(
        att_opt_net, X_tr_n, Y_att_tr, X_val_n, Y_att_val,
        lr=1e-3, max_epochs=200, patience=10, name="ATTnet_opt"
    )

    att_ckpt = os.path.join(CKPT_DIR, "ATTnet_opt_4PLD.pth")
    torch.save(att_opt_net.state_dict(), att_ckpt)
    print(f"Saved ATTnet_opt checkpoint to: {att_ckpt}")
    del Y_att_tr, Y_att_val
    memory_config.cleanup_memory()

    # 5. Train CBFnet_opt (5 hidden layers x 128 neurons, MSE loss)
    print("\n" + "=" * 60)
    print("Training CBFnet_opt (5x128, MSE loss, Optimized PLDs)...")
    print("=" * 60)
    cbf_opt_net = StandardizedNet(4, n_hidden_layers=5, n_neurons=128,
                                  y_mean=cbf_mean, y_std=cbf_std,
                                  y_min=physics.CBF_MIN, y_max=physics.CBF_MAX).to(device)

    Y_cbf_tr = y_tr_mmap[:, 0]
    Y_cbf_val = y_val_mmap[:, 0]
    cbf_opt_net, cbf_hist, cbf_time = train_standardized_model(
        cbf_opt_net, X_tr_n, Y_cbf_tr, X_val_n, Y_cbf_val,
        lr=1e-3, max_epochs=200, patience=10, name="CBFnet_opt"
    )

    cbf_ckpt = os.path.join(CKPT_DIR, "CBFnet_opt_4PLD.pth")
    torch.save(cbf_opt_net.state_dict(), cbf_ckpt)
    print(f"Saved CBFnet_opt checkpoint to: {cbf_ckpt}")

    # 6. Fit 5-Seed Ensemble Feature ELM on Optimized Latents
    print("\n" + "=" * 60)
    print("Fitting Ensemble Feature ELM (5 seeds) on Optimized Latents...")
    print("=" * 60)
    rng_elm = np.random.default_rng(2026)
    idx_elm = rng_elm.choice(1_000_000, size=100_000, replace=False)

    X_tr_sub = X_tr_n[idx_elm]
    Y_cbf_sub_std = ((y_tr_mmap[idx_elm, 0] - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_sub_std = ((y_tr_mmap[idx_elm, 1] - att_mean) / att_std).astype(np.float32)

    F_cbf_tr = cbf_opt_net.penultimate_features(X_tr_sub)
    F_att_tr = att_opt_net.penultimate_features(X_tr_sub)

    # Save ELM weights / fit metadata
    elm_seeds = [42, 43, 44, 45, 46]
    cbf_elms = []
    att_elms = []

    for s in elm_seeds:
        e_c = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_c.fit_chunked(F_cbf_tr, Y_cbf_sub_std)
        cbf_elms.append(e_c)

        e_a = GPUChunkedELM(n_hidden=512, alpha=1e-2, seed=s)
        e_a.fit_chunked(F_att_tr, Y_att_sub_std)
        att_elms.append(e_a)
    print(f"Fitted {len(elm_seeds)} ELM seeds for CBF and ATT.")

    # 7. Quick Validation Evaluation
    print("\n" + "=" * 60)
    print("Validation Set Quick Check (N=100,000)...")
    print("=" * 60)
    val_pred_att_dnn = att_opt_net.predict_physical(X_val_n)
    val_pred_cbf_dnn = cbf_opt_net.predict_physical(X_val_n)

    F_cbf_val = cbf_opt_net.penultimate_features(X_val_n)
    F_att_val = att_opt_net.penultimate_features(X_val_n)

    val_pred_cbf_elm = np.zeros(len(X_val_n), dtype=np.float32)
    val_pred_att_elm = np.zeros(len(X_val_n), dtype=np.float32)
    for i in range(len(elm_seeds)):
        val_pred_cbf_elm += np.clip(cbf_elms[i].predict(F_cbf_val) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
        val_pred_att_elm += np.clip(att_elms[i].predict(F_att_val) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
    val_pred_cbf_elm /= len(elm_seeds)
    val_pred_att_elm /= len(elm_seeds)

    m_cbf_dnn = calc_metrics(y_val_mmap[:, 0], val_pred_cbf_dnn)
    m_att_dnn = calc_metrics(y_val_mmap[:, 1], val_pred_att_dnn)
    m_cbf_elm = calc_metrics(y_val_mmap[:, 0], val_pred_cbf_elm)
    m_att_elm = calc_metrics(y_val_mmap[:, 1], val_pred_att_elm)

    print(f"Optimized DNN Val: CBF RMSE = {m_cbf_dnn['RMSE']:.4f}, MAE = {m_cbf_dnn['MAE']:.4f} | ATT RMSE = {m_att_dnn['RMSE']:.4f}s, MAE = {m_att_dnn['MAE']:.4f}s")
    print(f"Optimized ELM Val: CBF RMSE = {m_cbf_elm['RMSE']:.4f}, MAE = {m_cbf_elm['MAE']:.4f} | ATT RMSE = {m_att_elm['RMSE']:.4f}s, MAE = {m_att_elm['MAE']:.4f}s")

    # Early ATT validation check ([0.5, 1.0) s)
    val_early_mask = (y_val_mmap[:, 1] >= 0.5) & (y_val_mmap[:, 1] < 1.0)
    m_early_att_dnn = calc_metrics(y_val_mmap[val_early_mask, 1], val_pred_att_dnn[val_early_mask])
    m_early_att_elm = calc_metrics(y_val_mmap[val_early_mask, 1], val_pred_att_elm[val_early_mask])

    print(f"\nEARLY ATT [0.5, 1.0)s Val Performance (N={int(np.sum(val_early_mask))}):")
    print(f"  Optimized DNN: RMSE = {m_early_att_dnn['RMSE']:.4f}s, Bias = {m_early_att_dnn['Bias']:+.4f}s, CCC = {m_early_att_dnn['CCC']:.4f}")
    print(f"  Optimized ELM: RMSE = {m_early_att_elm['RMSE']:.4f}s, Bias = {m_early_att_elm['Bias']:+.4f}s, CCC = {m_early_att_elm['CCC']:.4f}")
    print(f"  (Reference Baseline A Early ATT on Val was RMSE ~0.466s, Bias ~+0.369s, CCC ~0.005!)")

    # Save training history
    train_history = {
        "plds": OPTIMIZED_PLDS.tolist(),
        "att_train_time_sec": att_time,
        "cbf_train_time_sec": cbf_time,
        "att_history": att_hist,
        "cbf_history": cbf_hist,
        "val_cbf_dnn": m_cbf_dnn,
        "val_att_dnn": m_att_dnn,
        "val_cbf_elm": m_cbf_elm,
        "val_att_elm": m_att_elm,
        "val_early_att_dnn": m_early_att_dnn,
        "val_early_att_elm": m_early_att_elm
    }

    import json
    with open(os.path.join(CKPT_DIR, "training_history_opt.json"), "w") as f:
        json.dump(train_history, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 5 TRAINING AND MODEL SELECTION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()

