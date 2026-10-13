"""
run_stage3_improvement_attempt.py
Stage 3: One Controlled Final Improvement Attempt for Experiment 5.
Evaluates at most two low-complexity, mathematically justified candidates:
- Candidate 1: Optimal Variance-Shrinkage Convex Blend (w * ELM + (1 - w) * DNN)
- Candidate 2: Deterministic Ridge-Regularized Latent Linear Readout (Linear Ridge on Penultimate)

Protocol:
1. Fit all models on training split (N=1,000,000) strictly.
2. Tune hyperparameters (w, alpha) on validation split (N=100,000) strictly.
3. Compare on validation split against Baseline DNN and Historical Feature ELM.
4. Freeze selected models, then evaluate on locked test cohort (N=100,000).
5. Output candidate_experiments.csv.
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

print(f"[Stage 3 Improvement] Device: {device}")

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

def solve_linear_ridge(F_tr, Y_tr_std, alpha_grid, F_val, Y_val_phys, y_std, y_mean, y_min, y_max):
    """Accumulate FtF and FtY in float64, find best alpha on validation set."""
    d = F_tr.shape[1]
    FtF = torch.zeros((d, d), dtype=torch.float64, device=device)
    FtY = torch.zeros((d, 1), dtype=torch.float64, device=device)
    
    chunk_size = 25_000
    N = F_tr.shape[0]
    n_chunks = int(np.ceil(N / chunk_size))
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)
        fc = torch.from_numpy(F_tr[start:end].astype(np.float32)).to(device)
        yc = torch.from_numpy(Y_tr_std[start:end].astype(np.float32)).to(device).unsqueeze(1)
        FtF += (fc.T @ fc).to(torch.float64)
        FtY += (fc.T @ yc).to(torch.float64)
        del fc, yc

    best_rmse = float("inf")
    best_alpha = None
    best_beta = None
    best_preds_val = None

    for alpha in alpha_grid:
        I = torch.eye(d, dtype=torch.float64, device=device)
        A = FtF + float(alpha) * I
        beta = torch.linalg.solve(A, FtY).to(torch.float32)
        
        # Predict validation
        preds_std = np.empty(F_val.shape[0], dtype=np.float32)
        n_val_chunks = int(np.ceil(F_val.shape[0] / chunk_size))
        with torch.no_grad():
            for c in range(n_val_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, F_val.shape[0])
                fvc = torch.from_numpy(F_val[start:end].astype(np.float32)).to(device)
                preds_std[start:end] = (fvc @ beta).squeeze(-1).cpu().numpy()
                del fvc
        
        preds_phys = np.clip(preds_std * y_std + y_mean, y_min, y_max)
        rmse = np.sqrt(np.mean((preds_phys - Y_val_phys) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_beta = beta
            best_preds_val = preds_phys
        del A, I

    return best_alpha, best_beta, best_preds_val

def main():
    print("=" * 80)
    print("STAGE 3: CONTROLLED FINAL IMPROVEMENT ATTEMPT")
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

    # Load Arrays
    print("Loading data splits...")
    X_tr_raw = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    Y_tr_raw = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    X_val_raw = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    Y_val_raw = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')
    X_te_raw = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    Y_te_raw = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')

    X_tr_std = (X_tr_raw[:] - X_mean) / X_std
    X_val_std = (X_val_raw[:] - X_mean) / X_std
    X_te_std = (X_te_raw[:] - X_mean) / X_std

    Y_cbf_tr_std = (Y_tr_raw[:, 0] - cbf_mean) / cbf_std
    Y_att_tr_std = (Y_tr_raw[:, 1] - att_mean) / att_std

    Y_cbf_val = Y_val_raw[:, 0].astype(np.float64)
    Y_att_val = Y_val_raw[:, 1].astype(np.float64)

    Y_cbf_te = Y_te_raw[:, 0].astype(np.float64)
    Y_att_te = Y_te_raw[:, 1].astype(np.float64)

    # Base DNN predictions
    pred_cbf_dnn_val = cbf_net.predict_physical(X_val_std)
    pred_att_dnn_val = att_net.predict_physical(X_val_std)
    pred_cbf_dnn_te = cbf_net.predict_physical(X_te_std)
    pred_att_dnn_te = att_net.predict_physical(X_te_std)

    # Extract Penultimate features
    print("Extracting penultimate features...")
    F_cbf_tr = extract_penultimate_features(cbf_net, X_tr_std)
    F_cbf_val = extract_penultimate_features(cbf_net, X_val_std)
    F_cbf_te = extract_penultimate_features(cbf_net, X_te_std)

    F_att_tr = extract_penultimate_features(att_net, X_tr_std)
    F_att_val = extract_penultimate_features(att_net, X_val_std)
    F_att_te = extract_penultimate_features(att_net, X_te_std)

    # Train Historical Feature ELM (5 seeds)
    print("Training Historical Feature ELM (5 seeds)...")
    def run_elm_ensemble(F_train, Y_std_tr, F_eval, y_std, y_mean, y_min, y_max, seeds=[42, 43, 44, 45, 46]):
        preds_sum = np.zeros(F_eval.shape[0], dtype=np.float64)
        for s in seeds:
            elm = GPUChunkedELM(n_hidden=512, alpha=1e-2, activation="tanh", seed=s)
            elm.fit_chunked(F_train, Y_std_tr, chunk_size=25_000)
            p_std = elm.predict(F_eval, chunk_size=25_000)
            preds_sum += np.clip(p_std * y_std + y_mean, y_min, y_max)
            del elm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return preds_sum / len(seeds)

    pred_cbf_elm_val = run_elm_ensemble(F_cbf_tr, Y_cbf_tr_std, F_cbf_val, cbf_std, cbf_mean, CBF_MIN, CBF_MAX)
    pred_att_elm_val = run_elm_ensemble(F_att_tr, Y_att_tr_std, F_att_val, att_std, att_mean, ATT_MIN, ATT_MAX)
    pred_cbf_elm_te = run_elm_ensemble(F_cbf_tr, Y_cbf_tr_std, F_cbf_te, cbf_std, cbf_mean, CBF_MIN, CBF_MAX)
    pred_att_elm_te = run_elm_ensemble(F_att_tr, Y_att_tr_std, F_att_te, att_std, att_mean, ATT_MIN, ATT_MAX)

    # -------------------------------------------------------------
    # CANDIDATE 1: Optimal Variance-Shrinkage Convex Ensemble Blend
    # w * ELM + (1 - w) * DNN, where w in [0, 1] is tuned strictly on validation
    # -------------------------------------------------------------
    print("\n--- Tuning Candidate 1: Convex Ensemble Blend on Validation Split ---")
    weights = np.linspace(0.0, 1.0, 101)
    
    # CBF weight optimization
    best_w_cbf = 1.0
    best_cbf_rmse_val = float("inf")
    for w in weights:
        p_blend = (1.0 - w) * pred_cbf_dnn_val + w * pred_cbf_elm_val
        rmse = np.sqrt(np.mean((p_blend - Y_cbf_val) ** 2))
        if rmse < best_cbf_rmse_val:
            best_cbf_rmse_val = rmse
            best_w_cbf = w

    # ATT weight optimization
    best_w_att = 1.0
    best_att_rmse_val = float("inf")
    for w in weights:
        p_blend = (1.0 - w) * pred_att_dnn_val + w * pred_att_elm_val
        rmse = np.sqrt(np.mean((p_blend - Y_att_val) ** 2))
        if rmse < best_att_rmse_val:
            best_att_rmse_val = rmse
            best_w_att = w

    print(f"  Optimal CBF Blend Weight: {best_w_cbf:.2f} * ELM + {1-best_w_cbf:.2f} * DNN | Val RMSE: {best_cbf_rmse_val:.4f}")
    print(f"  Optimal ATT Blend Weight: {best_w_att:.2f} * ELM + {1-best_w_att:.2f} * DNN | Val RMSE: {best_att_rmse_val:.4f}")

    pred_cbf_blend_val = (1.0 - best_w_cbf) * pred_cbf_dnn_val + best_w_cbf * pred_cbf_elm_val
    pred_att_blend_val = (1.0 - best_w_att) * pred_att_dnn_val + best_w_att * pred_att_elm_val
    pred_cbf_blend_te = (1.0 - best_w_cbf) * pred_cbf_dnn_te + best_w_cbf * pred_cbf_elm_te
    pred_att_blend_te = (1.0 - best_w_att) * pred_att_dnn_te + best_w_att * pred_att_elm_te

    # -------------------------------------------------------------
    # CANDIDATE 2: Ridge-Regularized Latent Linear Readout
    # Direct ridge regression on penultimate features (no random projection)
    # -------------------------------------------------------------
    print("\n--- Tuning Candidate 2: Latent Linear Ridge Readout on Validation Split ---")
    alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha_cbf, best_beta_cbf, pred_cbf_ridge_val = solve_linear_ridge(
        F_cbf_tr, Y_cbf_tr_std, alpha_grid, F_cbf_val, Y_cbf_val, cbf_std, cbf_mean, CBF_MIN, CBF_MAX
    )
    best_alpha_att, best_beta_att, pred_att_ridge_val = solve_linear_ridge(
        F_att_tr, Y_att_tr_std, alpha_grid, F_att_val, Y_att_val, att_std, att_mean, ATT_MIN, ATT_MAX
    )
    print(f"  Optimal CBF Linear Ridge Alpha: {best_alpha_cbf:.1e} | Val RMSE: {np.sqrt(np.mean((pred_cbf_ridge_val - Y_cbf_val)**2)):.4f}")
    print(f"  Optimal ATT Linear Ridge Alpha: {best_alpha_att:.1e} | Val RMSE: {np.sqrt(np.mean((pred_att_ridge_val - Y_att_val)**2)):.4f}")

    # Predict test for Linear Ridge
    with torch.no_grad():
        f_te_t_cbf = torch.from_numpy(F_cbf_te.astype(np.float32)).to(device)
        pred_cbf_ridge_te = np.clip((f_te_t_cbf @ best_beta_cbf).squeeze(-1).cpu().numpy() * cbf_std + cbf_mean, CBF_MIN, CBF_MAX)
        del f_te_t_cbf

        f_te_t_att = torch.from_numpy(F_att_te.astype(np.float32)).to(device)
        pred_att_ridge_te = np.clip((f_te_t_att @ best_beta_att).squeeze(-1).cpu().numpy() * att_std + att_mean, ATT_MIN, ATT_MAX)
        del f_te_t_att

    # -------------------------------------------------------------
    # Compile Comparative Validation & Test Results
    # -------------------------------------------------------------
    models_eval = {
        "Baseline A (DNN-G)": (pred_cbf_dnn_val, pred_att_dnn_val, pred_cbf_dnn_te, pred_att_dnn_te, "Baseline Frozen DNN"),
        "Historical Feature ELM (5 seeds)": (pred_cbf_elm_val, pred_att_elm_val, pred_cbf_elm_te, pred_att_elm_te, "Penultimate ELM (H=512, alpha=1e-2, 5 seeds)"),
        "Candidate 1: Convex Ensemble Blend": (pred_cbf_blend_val, pred_att_blend_val, pred_cbf_blend_te, pred_att_blend_te, f"Convex blend: {best_w_cbf:.2f} ELM + {1-best_w_cbf:.2f} DNN (CBF), {best_w_att:.2f} ELM + {1-best_w_att:.2f} DNN (ATT)"),
        "Candidate 2: Latent Linear Ridge": (pred_cbf_ridge_val, pred_att_ridge_val, pred_cbf_ridge_te, pred_att_ridge_te, f"Linear ridge on penultimate features (alpha={best_alpha_cbf:g}/{best_alpha_att:g})")
    }

    cbf_dnn_val_rmse = calc_metrics(Y_cbf_val, pred_cbf_dnn_val)["RMSE"]
    att_dnn_val_rmse = calc_metrics(Y_att_val, pred_att_dnn_val)["RMSE"]
    cbf_dnn_te_rmse = calc_metrics(Y_cbf_te, pred_cbf_dnn_te)["RMSE"]
    att_dnn_te_rmse = calc_metrics(Y_att_te, pred_att_dnn_te)["RMSE"]

    early_val_mask = (Y_att_val >= 0.5) & (Y_att_val < 1.0)
    early_te_mask = (Y_att_te >= 0.5) & (Y_att_te < 1.0)

    rows = []
    for name, (pcbf_v, patt_v, pcbf_t, patt_t, desc) in models_eval.items():
        mv_cbf = calc_metrics(Y_cbf_val, pcbf_v)
        mv_att = calc_metrics(Y_att_val, patt_v)
        mv_early = calc_metrics(Y_att_val[early_val_mask], patt_v[early_val_mask])

        mt_cbf = calc_metrics(Y_cbf_te, pcbf_t)
        mt_att = calc_metrics(Y_att_te, patt_t)
        mt_early = calc_metrics(Y_att_te[early_te_mask], patt_t[early_te_mask])

        cbf_val_gain = (cbf_dnn_val_rmse - mv_cbf["RMSE"]) / cbf_dnn_val_rmse * 100.0
        att_val_gain = (att_dnn_val_rmse - mv_att["RMSE"]) / att_dnn_val_rmse * 100.0

        cbf_te_gain = (cbf_dnn_te_rmse - mt_cbf["RMSE"]) / cbf_dnn_te_rmse * 100.0
        att_te_gain = (att_dnn_te_rmse - mt_att["RMSE"]) / att_dnn_te_rmse * 100.0

        rows.append({
            "Candidate_Name": name,
            "Description": desc,
            "Val_CBF_RMSE": round(mv_cbf["RMSE"], 4),
            "Val_CBF_Gain_%": round(cbf_val_gain, 2),
            "Val_ATT_RMSE": round(mv_att["RMSE"], 4),
            "Val_ATT_Gain_%": round(att_val_gain, 2),
            "Val_Early_ATT_RMSE": round(mv_early["RMSE"], 4),
            "Test_CBF_RMSE": round(mt_cbf["RMSE"], 4),
            "Test_CBF_Gain_%": round(cbf_te_gain, 2),
            "Test_ATT_RMSE": round(mt_att["RMSE"], 4),
            "Test_ATT_Gain_%": round(att_te_gain, 2),
            "Test_Early_ATT_RMSE": round(mt_early["RMSE"], 4)
        })

    cand_df = pd.DataFrame(rows)
    cand_csv = os.path.join(OUT_DIR, "candidate_experiments.csv")
    cand_df.to_csv(cand_csv, index=False)
    print("\n--- Candidate Evaluation Table ---")
    print(cand_df[["Candidate_Name", "Val_CBF_RMSE", "Val_CBF_Gain_%", "Val_ATT_RMSE", "Val_ATT_Gain_%", "Test_CBF_RMSE", "Test_CBF_Gain_%", "Test_ATT_RMSE", "Test_ATT_Gain_%"]].to_string())

if __name__ == "__main__":
    main()

