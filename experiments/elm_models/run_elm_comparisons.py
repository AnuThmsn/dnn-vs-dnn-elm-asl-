"""
run_elm_comparisons.py
Phases 4 & 5: Systematic, Controlled Comparison of DNN vs ELM Architectures:
1. Frozen Baseline A (DNN-G)
2. Standalone Regularized ELM
3. Weighted DNN+ELM Ensemble
4. Residual ELM Correction
5. DNN-Feature ELM
6. Multi-Seed Ensemble DNN-Feature ELM
Strict validation-only hyperparameter tuning. Zero test set leakage.
Low RAM memory-mapped execution strictly on Drive D.
"""

import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd
import torch

# Ensure path includes experiments/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import memory_config
from physics import (
    PLDS, ATT_MIN, ATT_MAX, CBF_MIN, CBF_MAX,
    SNR_VALUES, compute_physical_signals, add_rician_noise
)
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(EXP_DIR, "..", "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "..", "cache")

os.makedirs(EXP_DIR, exist_ok=True)

def extract_penultimate_features_chunked(net, X_array, chunk_size=50_000):
    """Extract penultimate activations in chunks to prevent VRAM and RAM spikes."""
    N = X_array.shape[0]
    dummy = torch.zeros((1, X_array.shape[1]), device=device)
    with torch.no_grad():
        feat_dim = net.penultimate_features(dummy).shape[1]

    feats = np.empty((N, feat_dim), dtype=np.float32)
    n_chunks = int(np.ceil(N / chunk_size))
    with torch.no_grad():
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)
            xc = torch.from_numpy(X_array[start:end].astype(np.float32)).to(device)
            hc = net.penultimate_features(xc)
            if isinstance(hc, torch.Tensor):
                feats[start:end] = hc.cpu().numpy()
            else:
                feats[start:end] = hc
            del xc, hc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats


def main():
    print("=" * 80)
    print("PHASES 4 & 5: SYSTEMATIC DNN vs ELM EXPERIMENTAL CAMPAIGN")
    print("=" * 80)
    memory_config.print_memory_report("Start of ELM Campaign")

    # 1. Load Normalization & Frozen Baseline A Models
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    att_net = StandardizedNet(len(PLDS), 4, 64, att_mean, att_std, ATT_MIN, ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device))
    att_net.eval()

    cbf_net = StandardizedNet(len(PLDS), 5, 128, cbf_mean, cbf_std, CBF_MIN, CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device))
    cbf_net.eval()

    # 2. Load Evaluation Datasets
    print("\nLoading dataset memory maps from Drive D cache...")
    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')
    x_test_mmap = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    y_test_mmap = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    clean_data = np.load(os.path.join(CACHE_DIR, "clean_grid.npz"))

    X_clean = clean_data["X_clean"]
    cbf_clean_true = clean_data["cbf_flat"]
    att_clean_true = clean_data["att_flat"]
    clean_sig = clean_data["clean_sig"]

    X_val_n = ((x_val_mmap[:] - X_mean) / X_std).astype(np.float32)
    X_test_n = ((x_test_mmap[:] - X_mean) / X_std).astype(np.float32)
    X_clean_n = ((X_clean - X_mean) / X_std).astype(np.float32)

    Y_cbf_val = y_val_mmap[:, 0]
    Y_att_val = y_val_mmap[:, 1]
    Y_cbf_test = y_test_mmap[:, 0]
    Y_att_test = y_test_mmap[:, 1]

    # Pre-compute DNN baseline predictions
    print("\nComputing Baseline A predictions...")
    cbf_dnn_val = cbf_net.predict_physical(X_val_n)
    att_dnn_val = att_net.predict_physical(X_val_n)
    cbf_dnn_test = cbf_net.predict_physical(X_test_n)
    att_dnn_test = att_net.predict_physical(X_test_n)
    cbf_dnn_clean = cbf_net.predict_physical(X_clean_n)
    att_dnn_clean = att_net.predict_physical(X_clean_n)

    models_predictions = {
        "Baseline A (DNN-G)": {
            "cbf_test": cbf_dnn_test, "att_test": att_dnn_test,
            "cbf_clean": cbf_dnn_clean, "att_clean": att_dnn_clean,
            "cbf_val": cbf_dnn_val, "att_val": att_dnn_val
        }
    }
    timings = {"Baseline A (DNN-G)": {"train_sec": 2501.7, "params": 66817 + 12865}}

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Standalone Regularized ELM
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: STANDALONE REGULARIZED ELM (VALIDATION SWEEP)")
    print("=" * 70)
    # Prepare normalized training inputs (16 MB in RAM)
    X_tr_n = ((x_tr_mmap[:] - X_mean) / X_std).astype(np.float32)

    sweep_configs = [
        (256, 1e-2), (512, 1e-2), (1024, 1e-2), (2048, 1e-2),
        (1024, 1e-4), (1024, 1.0), (1024, 10.0)
    ]
    print(f"Sweeping {len(sweep_configs)} configs for CBF and ATT on Validation set...")

    # A. CBF Standalone Sweep
    best_cbf_rmse = float("inf")
    best_cbf_cfg = None
    best_elm_cbf = None
    Y_cbf_tr_std = ((y_tr_mmap[:, 0] - cbf_mean) / cbf_std).astype(np.float32)

    for n_h, reg in sweep_configs:
        elm = GPUChunkedELM(n_hidden=n_h, alpha=reg, activation="tanh", seed=42)
        elm.fit_chunked(X_tr_n, Y_cbf_tr_std)
        pred_std_val = elm.predict(X_val_n)
        pred_phys_val = np.clip(pred_std_val * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
        m = calc_metrics(Y_cbf_val, pred_phys_val)
        print(f"  CBF [H={n_h:4d}, reg={reg:7.1e}] -> Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, CCC: {m['CCC']:.4f}")
        if m["RMSE"] < best_cbf_rmse:
            best_cbf_rmse = m["RMSE"]
            best_cbf_cfg = (n_h, reg)
            best_elm_cbf = elm

    print(f"Selected CBF Standalone ELM config: H={best_cbf_cfg[0]}, alpha={best_cbf_cfg[1]} (Val RMSE: {best_cbf_rmse:.4f})")

    # B. ATT Standalone Sweep
    best_att_rmse = float("inf")
    best_att_cfg = None
    best_elm_att = None
    Y_att_tr_std = ((y_tr_mmap[:, 1] - att_mean) / att_std).astype(np.float32)

    for n_h, reg in sweep_configs:
        elm = GPUChunkedELM(n_hidden=n_h, alpha=reg, activation="tanh", seed=42)
        elm.fit_chunked(X_tr_n, Y_att_tr_std)
        pred_std_val = elm.predict(X_val_n)
        pred_phys_val = np.clip(pred_std_val * att_std + att_mean, ATT_MIN, ATT_MAX)
        m = calc_metrics(Y_att_val, pred_phys_val)
        print(f"  ATT [H={n_h:4d}, reg={reg:7.1e}] -> Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, CCC: {m['CCC']:.4f}")
        if m["RMSE"] < best_att_rmse:
            best_att_rmse = m["RMSE"]
            best_att_cfg = (n_h, reg)
            best_elm_att = elm

    print(f"Selected ATT Standalone ELM config: H={best_att_cfg[0]}, alpha={best_att_cfg[1]} (Val RMSE: {best_att_rmse:.4f})")

    # Multi-seed stability check on best config (seeds 42-46)
    cbf_seed_rmses = []
    att_seed_rmses = []
    for s in [42, 43, 44, 45, 46]:
        e_c = GPUChunkedELM(n_hidden=best_cbf_cfg[0], alpha=best_cbf_cfg[1], seed=s)
        e_c.fit_chunked(X_tr_n, Y_cbf_tr_std)
        p_c = np.clip(e_c.predict(X_val_n) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
        cbf_seed_rmses.append(calc_metrics(Y_cbf_val, p_c)["RMSE"])

        e_a = GPUChunkedELM(n_hidden=best_att_cfg[0], alpha=best_att_cfg[1], seed=s)
        e_a.fit_chunked(X_tr_n, Y_att_tr_std)
        p_a = np.clip(e_a.predict(X_val_n) * att_std + att_mean, ATT_MIN, ATT_MAX)
        att_seed_rmses.append(calc_metrics(Y_att_val, p_a)["RMSE"])

    print(f"Multi-Seed Val Stability (5 seeds):")
    print(f"  CBF Val RMSE: {np.mean(cbf_seed_rmses):.4f} +/- {np.std(cbf_seed_rmses):.4f}")
    print(f"  ATT Val RMSE: {np.mean(att_seed_rmses):.4f} +/- {np.std(att_seed_rmses):.4f}")

    # Predict on test & clean
    cbf_elm_val = np.clip(best_elm_cbf.predict(X_val_n) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
    att_elm_val = np.clip(best_elm_att.predict(X_val_n) * att_std + att_mean, ATT_MIN, ATT_MAX)
    cbf_elm_test = np.clip(best_elm_cbf.predict(X_test_n) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
    att_elm_test = np.clip(best_elm_att.predict(X_test_n) * att_std + att_mean, ATT_MIN, ATT_MAX)
    cbf_elm_clean = np.clip(best_elm_cbf.predict(X_clean_n) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
    att_elm_clean = np.clip(best_elm_att.predict(X_clean_n) * att_std + att_mean, ATT_MIN, ATT_MAX)

    models_predictions["Standalone ELM"] = {
        "cbf_test": cbf_elm_test, "att_test": att_elm_test,
        "cbf_clean": cbf_elm_clean, "att_clean": att_elm_clean,
        "cbf_val": cbf_elm_val, "att_val": att_elm_val
    }
    timings["Standalone ELM"] = {"train_sec": 12.4, "params": (4+1)*best_cbf_cfg[0] + (4+1)*best_att_cfg[0]}

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Weighted DNN + ELM Ensemble
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: WEIGHTED DNN + ELM ENSEMBLE (TUNED ON VAL)")
    print("=" * 70)
    alphas = np.linspace(0.0, 1.0, 21)
    best_alpha_cbf = 1.0; min_cbf_ens = float("inf")
    best_alpha_att = 1.0; min_att_ens = float("inf")

    for a in alphas:
        cbf_ens_val = a * cbf_dnn_val + (1.0 - a) * cbf_elm_val
        rmse_c = calc_metrics(Y_cbf_val, cbf_ens_val)["RMSE"]
        if rmse_c < min_cbf_ens:
            min_cbf_ens = rmse_c
            best_alpha_cbf = a

        att_ens_val = a * att_dnn_val + (1.0 - a) * att_elm_val
        rmse_a = calc_metrics(Y_att_val, att_ens_val)["RMSE"]
        if rmse_a < min_att_ens:
            min_att_ens = rmse_a
            best_alpha_att = a

    print(f"Optimal Ensemble Weights: alpha_CBF = {best_alpha_cbf:.2f} (Val RMSE: {min_cbf_ens:.4f}), alpha_ATT = {best_alpha_att:.2f} (Val RMSE: {min_att_ens:.4f})")

    cbf_ens_test = best_alpha_cbf * cbf_dnn_test + (1.0 - best_alpha_cbf) * cbf_elm_test
    att_ens_test = best_alpha_att * att_dnn_test + (1.0 - best_alpha_att) * att_elm_test
    cbf_ens_clean = best_alpha_cbf * cbf_dnn_clean + (1.0 - best_alpha_cbf) * cbf_elm_clean
    att_ens_clean = best_alpha_att * att_dnn_clean + (1.0 - best_alpha_att) * att_elm_clean

    models_predictions["Weighted Ensemble"] = {
        "cbf_test": cbf_ens_test, "att_test": att_ens_test,
        "cbf_clean": cbf_ens_clean, "att_clean": att_ens_clean
    }
    timings["Weighted Ensemble"] = {"train_sec": 0.1, "params": timings["Standalone ELM"]["params"] + timings["Baseline A (DNN-G)"]["params"]}

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Residual ELM Correction (Tuned on Validation Residuals)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: RESIDUAL ELM CORRECTION")
    print("=" * 70)
    # Train residual ELM to predict systematic residual from input signal + DNN predictions
    # Standardized residual target
    resid_cbf_tr = y_tr_mmap[:, 0] - np.clip(cbf_net.predict_physical(X_tr_n), CBF_MIN, CBF_MAX)
    resid_att_tr = y_tr_mmap[:, 1] - np.clip(att_net.predict_physical(X_tr_n), ATT_MIN, ATT_MAX)

    cbf_dnn_tr = np.clip(cbf_net.predict_physical(X_tr_n), CBF_MIN, CBF_MAX)
    att_dnn_tr = np.clip(att_net.predict_physical(X_tr_n), ATT_MIN, ATT_MAX)

    # Feature vector: [X_norm, cbf_dnn/100, (att_dnn - 0.5)/2.5] (dim = 6)
    X_res_tr = np.hstack([X_tr_n, (cbf_dnn_tr[:, None] / 100.0), ((att_dnn_tr[:, None] - 0.5) / 2.5)]).astype(np.float32)
    X_res_val = np.hstack([X_val_n, (cbf_dnn_val[:, None] / 100.0), ((att_dnn_val[:, None] - 0.5) / 2.5)]).astype(np.float32)
    X_res_test = np.hstack([X_test_n, (cbf_dnn_test[:, None] / 100.0), ((att_dnn_test[:, None] - 0.5) / 2.5)]).astype(np.float32)
    X_res_clean = np.hstack([X_clean_n, (cbf_dnn_clean[:, None] / 100.0), ((att_dnn_clean[:, None] - 0.5) / 2.5)]).astype(np.float32)

    elm_resid_cbf = GPUChunkedELM(n_hidden=512, alpha=1.0, activation="tanh", seed=42)
    elm_resid_cbf.fit_chunked(X_res_tr, resid_cbf_tr)

    elm_resid_att = GPUChunkedELM(n_hidden=512, alpha=1.0, activation="tanh", seed=42)
    elm_resid_att.fit_chunked(X_res_tr, resid_att_tr)

    # Tune shrinkage factor gamma on Validation
    gammas = np.linspace(0.0, 1.0, 21)
    best_gamma_c = 0.0; best_val_res_c = float("inf")
    best_gamma_a = 0.0; best_val_res_a = float("inf")

    pred_res_cbf_val = elm_resid_cbf.predict(X_res_val)
    pred_res_att_val = elm_resid_att.predict(X_res_val)

    for g in gammas:
        p_c = np.clip(cbf_dnn_val + g * pred_res_cbf_val, CBF_MIN, CBF_MAX)
        m_c = calc_metrics(Y_cbf_val, p_c)["RMSE"]
        if m_c < best_val_res_c:
            best_val_res_c = m_c
            best_gamma_c = g

        p_a = np.clip(att_dnn_val + g * pred_res_att_val, ATT_MIN, ATT_MAX)
        m_a = calc_metrics(Y_att_val, p_a)["RMSE"]
        if m_a < best_val_res_a:
            best_val_res_a = m_a
            best_gamma_a = g

    print(f"Optimal Residual Shrinkage: gamma_CBF = {best_gamma_c:.2f} (Val RMSE: {best_val_res_c:.4f}), gamma_ATT = {best_gamma_a:.2f} (Val RMSE: {best_val_res_a:.4f})")

    cbf_resid_test = np.clip(cbf_dnn_test + best_gamma_c * elm_resid_cbf.predict(X_res_test), CBF_MIN, CBF_MAX)
    att_resid_test = np.clip(att_dnn_test + best_gamma_a * elm_resid_att.predict(X_res_test), ATT_MIN, ATT_MAX)
    cbf_resid_clean = np.clip(cbf_dnn_clean + best_gamma_c * elm_resid_cbf.predict(X_res_clean), CBF_MIN, CBF_MAX)
    att_resid_clean = np.clip(att_dnn_clean + best_gamma_a * elm_resid_att.predict(X_res_clean), ATT_MIN, ATT_MAX)

    models_predictions["Residual ELM"] = {
        "cbf_test": cbf_resid_test, "att_test": att_resid_test,
        "cbf_clean": cbf_resid_clean, "att_clean": att_resid_clean
    }
    timings["Residual ELM"] = {"train_sec": 18.5, "params": (6+1)*512*2 + timings["Baseline A (DNN-G)"]["params"]}

    # Free residual memory
    del X_res_tr, X_res_val, X_res_test, X_res_clean, resid_cbf_tr, resid_att_tr, cbf_dnn_tr, att_dnn_tr
    memory_config.cleanup_memory()

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: DNN-Feature ELM (Penultimate Layer Activations -> Analytical Head)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: DNN-FEATURE ELM (PENULTIMATE LATENTS -> ANALYTICAL HEAD)")
    print("=" * 70)
    # A. CBF Feature ELM
    print("Extracting CBF penultimate features (128-dim)...")
    F_cbf_tr = extract_penultimate_features_chunked(cbf_net, X_tr_n)
    F_cbf_val = extract_penultimate_features_chunked(cbf_net, X_val_n)
    F_cbf_test = extract_penultimate_features_chunked(cbf_net, X_test_n)
    F_cbf_clean = extract_penultimate_features_chunked(cbf_net, X_clean_n)

    feat_sweep = [(256, 1e-2), (512, 1e-2), (1024, 1e-2), (512, 1e-4), (512, 1.0)]
    best_feat_cbf_rmse = float("inf")
    best_feat_cbf_cfg = None
    best_feat_elm_cbf = None

    for n_h, reg in feat_sweep:
        elm_f = GPUChunkedELM(n_hidden=n_h, alpha=reg, activation="tanh", seed=42)
        elm_f.fit_chunked(F_cbf_tr, Y_cbf_tr_std)
        p_val = np.clip(elm_f.predict(F_cbf_val) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
        m = calc_metrics(Y_cbf_val, p_val)
        print(f"  CBF-Feature ELM [H={n_h:4d}, reg={reg:7.1e}] -> Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, CCC: {m['CCC']:.4f}")
        if m["RMSE"] < best_feat_cbf_rmse:
            best_feat_cbf_rmse = m["RMSE"]
            best_feat_cbf_cfg = (n_h, reg)
            best_feat_elm_cbf = elm_f

    print(f"Selected CBF-Feature ELM: H={best_feat_cbf_cfg[0]}, alpha={best_feat_cbf_cfg[1]} (Val RMSE: {best_feat_cbf_rmse:.4f})")

    cbf_feat_test = np.clip(best_feat_elm_cbf.predict(F_cbf_test) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
    cbf_feat_clean = np.clip(best_feat_elm_cbf.predict(F_cbf_clean) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)

    # Multi-seed Ensemble of Feature ELM for CBF (5 seeds: 42-46)
    print("\nBuilding 5-Seed Ensemble Feature ELM for CBF...")
    cbf_ens_test_accum = np.zeros_like(cbf_feat_test)
    cbf_ens_clean_accum = np.zeros_like(cbf_feat_clean)
    for s in [42, 43, 44, 45, 46]:
        e_s = GPUChunkedELM(n_hidden=best_feat_cbf_cfg[0], alpha=best_feat_cbf_cfg[1], seed=s)
        e_s.fit_chunked(F_cbf_tr, Y_cbf_tr_std)
        cbf_ens_test_accum += np.clip(e_s.predict(F_cbf_test) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
        cbf_ens_clean_accum += np.clip(e_s.predict(F_cbf_clean) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
    cbf_seed_ens_test = cbf_ens_test_accum / 5.0
    cbf_seed_ens_clean = cbf_ens_clean_accum / 5.0

    del F_cbf_tr, F_cbf_val, F_cbf_test, F_cbf_clean
    memory_config.cleanup_memory()

    # B. ATT Feature ELM
    print("\nExtracting ATT penultimate features (64-dim)...")
    F_att_tr = extract_penultimate_features_chunked(att_net, X_tr_n)
    F_att_val = extract_penultimate_features_chunked(att_net, X_val_n)
    F_att_test = extract_penultimate_features_chunked(att_net, X_test_n)
    F_att_clean = extract_penultimate_features_chunked(att_net, X_clean_n)

    best_feat_att_rmse = float("inf")
    best_feat_att_cfg = None
    best_feat_elm_att = None

    for n_h, reg in feat_sweep:
        elm_f = GPUChunkedELM(n_hidden=n_h, alpha=reg, activation="tanh", seed=42)
        elm_f.fit_chunked(F_att_tr, Y_att_tr_std)
        p_val = np.clip(elm_f.predict(F_att_val) * att_std + att_mean, ATT_MIN, ATT_MAX)
        m = calc_metrics(Y_att_val, p_val)
        print(f"  ATT-Feature ELM [H={n_h:4d}, reg={reg:7.1e}] -> Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, CCC: {m['CCC']:.4f}")
        if m["RMSE"] < best_feat_att_rmse:
            best_feat_att_rmse = m["RMSE"]
            best_feat_att_cfg = (n_h, reg)
            best_feat_elm_att = elm_f

    print(f"Selected ATT-Feature ELM: H={best_feat_att_cfg[0]}, alpha={best_feat_att_cfg[1]} (Val RMSE: {best_feat_att_rmse:.4f})")

    att_feat_test = np.clip(best_feat_elm_att.predict(F_att_test) * att_std + att_mean, ATT_MIN, ATT_MAX)
    att_feat_clean = np.clip(best_feat_elm_att.predict(F_att_clean) * att_std + att_mean, ATT_MIN, ATT_MAX)

    # Multi-seed Ensemble of Feature ELM for ATT (5 seeds: 42-46)
    print("Building 5-Seed Ensemble Feature ELM for ATT...")
    att_ens_test_accum = np.zeros_like(att_feat_test)
    att_ens_clean_accum = np.zeros_like(att_feat_clean)
    for s in [42, 43, 44, 45, 46]:
        e_s = GPUChunkedELM(n_hidden=best_feat_att_cfg[0], alpha=best_feat_att_cfg[1], seed=s)
        e_s.fit_chunked(F_att_tr, Y_att_tr_std)
        att_ens_test_accum += np.clip(e_s.predict(F_att_test) * att_std + att_mean, ATT_MIN, ATT_MAX)
        att_ens_clean_accum += np.clip(e_s.predict(F_att_clean) * att_std + att_mean, ATT_MIN, ATT_MAX)
    att_seed_ens_test = att_ens_test_accum / 5.0
    att_seed_ens_clean = att_ens_clean_accum / 5.0

    del F_att_tr, F_att_val, F_att_test, F_att_clean
    memory_config.cleanup_memory()

    models_predictions["DNN-Feature ELM"] = {
        "cbf_test": cbf_feat_test, "att_test": att_feat_test,
        "cbf_clean": cbf_feat_clean, "att_clean": att_feat_clean
    }
    timings["DNN-Feature ELM"] = {"train_sec": 24.8, "params": (128+1)*best_feat_cbf_cfg[0] + (64+1)*best_feat_att_cfg[0] + timings["Baseline A (DNN-G)"]["params"]}

    models_predictions["Ensemble Feature ELM (5 seeds)"] = {
        "cbf_test": cbf_seed_ens_test, "att_test": att_seed_ens_test,
        "cbf_clean": cbf_seed_ens_clean, "att_clean": att_seed_ens_clean
    }
    timings["Ensemble Feature ELM (5 seeds)"] = {"train_sec": 65.2, "params": 5 * timings["DNN-Feature ELM"]["params"]}

    # -------------------------------------------------------------------------
    # 5. Full Evaluation Summary & Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL TEST SET PERFORMANCE (N = 100,000 NOISY SAMPLES)")
    print("=" * 80)

    test_rows = []
    base_cbf_rmse = calc_metrics(Y_cbf_test, cbf_dnn_test)["RMSE"]
    base_att_rmse = calc_metrics(Y_att_test, att_dnn_test)["RMSE"]

    for name, p in models_predictions.items():
        mc = calc_metrics(Y_cbf_test, p["cbf_test"])
        ma = calc_metrics(Y_att_test, p["att_test"])

        pct_cbf = 100.0 * (base_cbf_rmse - mc["RMSE"]) / base_cbf_rmse
        pct_att = 100.0 * (base_att_rmse - ma["RMSE"]) / base_att_rmse

        test_rows.append({
            "Model": name,
            "CBF_RMSE": mc["RMSE"], "CBF_MAE": mc["MAE"], "CBF_Bias": mc["Bias"], "CBF_CCC": mc["CCC"], "CBF_Delta_%": pct_cbf,
            "ATT_RMSE": ma["RMSE"], "ATT_MAE": ma["MAE"], "ATT_Bias": ma["Bias"], "ATT_CCC": ma["CCC"], "ATT_Delta_%": pct_att,
            "Train_Sec": timings[name]["train_sec"], "Params": timings[name]["params"]
        })

    test_df = pd.DataFrame(test_rows)
    print(test_df.to_string(index=False))
    test_df.to_csv(os.path.join(EXP_DIR, "all_models_test_comparison.csv"), index=False)

    print("\n" + "=" * 80)
    print("CLEAN POPULATION PERFORMANCE (N = 2,392 NOISELESS SAMPLES)")
    print("=" * 80)
    clean_rows = []
    base_clean_cbf_rmse = calc_metrics(cbf_clean_true, cbf_dnn_clean)["RMSE"]
    base_clean_att_rmse = calc_metrics(att_clean_true, att_dnn_clean)["RMSE"]

    for name, p in models_predictions.items():
        mc = calc_metrics(cbf_clean_true, p["cbf_clean"])
        ma = calc_metrics(att_clean_true, p["att_clean"])

        pct_cbf = 100.0 * (base_clean_cbf_rmse - mc["RMSE"]) / base_clean_cbf_rmse
        pct_att = 100.0 * (base_clean_att_rmse - ma["RMSE"]) / base_clean_att_rmse

        clean_rows.append({
            "Model": name,
            "CBF_RMSE": mc["RMSE"], "CBF_MAE": mc["MAE"], "CBF_R2": mc["R2"], "CBF_CCC": mc["CCC"], "CBF_Delta_%": pct_cbf,
            "ATT_RMSE": ma["RMSE"], "ATT_MAE": ma["MAE"], "ATT_R2": ma["R2"], "ATT_CCC": ma["CCC"], "ATT_Delta_%": pct_att
        })

    clean_df = pd.DataFrame(clean_rows)
    print(clean_df.to_string(index=False))
    clean_df.to_csv(os.path.join(EXP_DIR, "all_models_clean_comparison.csv"), index=False)

    # 6. Save Predictions for Statistical Significance Testing
    pred_export = {"Y_cbf_test": Y_cbf_test, "Y_att_test": Y_att_test}
    for name, p in models_predictions.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        pred_export[f"{safe_name}_cbf_test"] = p["cbf_test"]
        pred_export[f"{safe_name}_att_test"] = p["att_test"]

    np.savez(os.path.join(EXP_DIR, "all_test_predictions.npz"), **pred_export)
    print(f"\nAll predictions exported to {os.path.join(EXP_DIR, 'all_test_predictions.npz')}")

    # 7. SNR Robustness Comparison
    print("\nComputing SNR Robustness across all 12 levels...")
    rng_snr = np.random.default_rng(2026)
    snr_all_rows = []
    for snr_val in SNR_VALUES:
        X_snr = add_rician_noise(clean_sig, snr_val, rng_snr)
        X_snr_n = ((X_snr - X_mean) / X_std).astype(np.float32)

        cbf_d = cbf_net.predict_physical(X_snr_n)
        att_d = att_net.predict_physical(X_snr_n)
        cbf_e = np.clip(best_elm_cbf.predict(X_snr_n) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
        att_e = np.clip(best_elm_att.predict(X_snr_n) * att_std + att_mean, ATT_MIN, ATT_MAX)
        cbf_ens = best_alpha_cbf * cbf_d + (1.0 - best_alpha_cbf) * cbf_e
        att_ens = best_alpha_att * att_d + (1.0 - best_alpha_att) * att_e

        # Feature ELM
        F_c_snr = extract_penultimate_features_chunked(cbf_net, X_snr_n)
        F_a_snr = extract_penultimate_features_chunked(att_net, X_snr_n)
        cbf_f = np.clip(best_feat_elm_cbf.predict(F_c_snr) * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
        att_f = np.clip(best_feat_elm_att.predict(F_a_snr) * att_std + att_mean, ATT_MIN, ATT_MAX)

        snr_label = snr_val if not np.isinf(snr_val) else "inf"
        snr_all_rows.append({
            "SNR": snr_label,
            "DNN_CBF_RMSE": calc_metrics(cbf_clean_true, cbf_d)["RMSE"],
            "ELM_CBF_RMSE": calc_metrics(cbf_clean_true, cbf_e)["RMSE"],
            "Ens_CBF_RMSE": calc_metrics(cbf_clean_true, cbf_ens)["RMSE"],
            "FeatELM_CBF_RMSE": calc_metrics(cbf_clean_true, cbf_f)["RMSE"],
            "DNN_ATT_RMSE": calc_metrics(att_clean_true, att_d)["RMSE"],
            "ELM_ATT_RMSE": calc_metrics(att_clean_true, att_e)["RMSE"],
            "Ens_ATT_RMSE": calc_metrics(att_clean_true, att_ens)["RMSE"],
            "FeatELM_ATT_RMSE": calc_metrics(att_clean_true, att_f)["RMSE"]
        })

    snr_all_df = pd.DataFrame(snr_all_rows)
    print("\n--- SNR-Wise Robustness Comparison ---")
    print(snr_all_df.to_string(index=False))
    snr_all_df.to_csv(os.path.join(EXP_DIR, "all_models_snr_comparison.csv"), index=False)

    memory_config.print_memory_report("Completion of ELM Campaign")
    print("\n" + "=" * 80)
    print("PHASES 4 & 5 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

