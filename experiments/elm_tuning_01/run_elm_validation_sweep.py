"""
run_elm_validation_sweep.py
Stage 2 & Stage 3: Leakage-Free ELM Hyperparameter Optimization on Validation Set.
- Extracts frozen penultimate features from Baseline A CBFnet (128-dim) and ATTnet (64-dim).
- Caches features as memory-mapped arrays on Drive D to strictly adhere to < 80 MB RAM safeguards.
- Systematically sweeps:
    * Hidden neuron counts: [128, 256, 512, 1024, 2048]
    * Regularization alpha: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    * Activations: ['tanh', 'elu', 'relu']
    * Random seeds: 10 seeds [42, 43, 44, 45, 46, 47, 48, 49, 50, 51] for top configurations.
- Fast ridge reuse: Computes H^T H and H^T Y once per (H, activation, seed), instantly solving across all alphas.
- Separate, decoupled optimization for CBF and ATT.
- Evaluates validation RMSE (primary criterion), MAE, CCC, and early-ATT diagnostic ([0.5, 1.0) s).
- Saves validation_sweep.csv, seed_variability.csv, and validation_comparison.csv immediately.
"""

import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd
import torch

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
OUT_DIR = os.path.join(EXP_DIR, "elm_tuning_01")
FEAT_CACHE_DIR = os.path.join(OUT_DIR, "cache")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

for d in [OUT_DIR, FEAT_CACHE_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.append(EXP_DIR)
sys.path.append(os.path.join(EXP_DIR, "elm_models"))
import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

class FastGPUChunkedELM(GPUChunkedELM):
    """
    Subclass that separates H^T H accumulation from the regularized solve,
    enabling instant multi-alpha evaluation without redundant 1M-sample passes.
    """
    def compute_HtH_HtY(self, X_array, y_array, chunk_size=50_000):
        N, d = X_array.shape
        if self.W is None:
            self._init_weights(d)

        L = self.n_hidden
        HtH = torch.zeros((L, L), dtype=torch.float64, device=device)
        HtY = torch.zeros((L, 1), dtype=torch.float64, device=device)

        n_chunks = int(np.ceil(N / chunk_size))
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)

            xc = torch.from_numpy(X_array[start:end].astype(np.float32)).to(device)
            yc = torch.from_numpy(y_array[start:end].astype(np.float32)).to(device)
            if yc.ndim == 1:
                yc = yc.unsqueeze(1)

            hc = self._activate(xc @ self.W + self.b).to(torch.float64)
            HtH += hc.T @ hc
            HtY += hc.T @ yc.to(torch.float64)
            del xc, yc, hc

        return HtH, HtY

    def solve_alpha(self, HtH, HtY, alpha):
        self.alpha = float(alpha)
        L = self.n_hidden
        I = torch.eye(L, dtype=torch.float64, device=device)
        A = HtH + self.alpha * I
        self.beta = torch.linalg.solve(A, HtY).to(torch.float32)
        del A, I
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fit_chunked(self, X_array, y_array, chunk_size=50_000):
        HtH, HtY = self.compute_HtH_HtY(X_array, y_array, chunk_size)
        self.solve_alpha(HtH, HtY, self.alpha)
        del HtH, HtY

def extract_features_to_mmap(net, X_array, out_path, chunk_size=50_000):
    """Extract penultimate activations in low-RAM chunks directly to disk."""
    N = X_array.shape[0]
    dummy = torch.zeros((1, X_array.shape[1]), device=device)
    with torch.no_grad():
        feat_dim = net.penultimate_features(dummy).shape[1]

    if os.path.exists(out_path) and os.path.getsize(out_path) == N * feat_dim * 4:
        print(f"Loading existing cached features from {out_path}...")
        return np.load(out_path, mmap_mode='r')

    print(f"Extracting {feat_dim}-dim features for N={N:,} to {out_path}...")
    f_mmap = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(N, feat_dim))

    n_chunks = int(np.ceil(N / chunk_size))
    with torch.no_grad():
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)
            xc = torch.from_numpy(X_array[start:end].astype(np.float32)).to(device)
            hc = net.penultimate_features(xc)
            if isinstance(hc, torch.Tensor):
                f_mmap[start:end] = hc.cpu().numpy()
            else:
                f_mmap[start:end] = hc
            f_mmap.flush()
            del xc, hc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del f_mmap
    gc.collect()
    return np.load(out_path, mmap_mode='r')

def main():
    print("=" * 80)
    print("STAGE 2 & 3: LEAKAGE-FREE ELM HYPERPARAMETER SWEEP ON VALIDATION SET")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Normalization & Checkpoints
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

    # 2. Memory-map raw inputs and targets
    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    X_val_n = ((x_val_mmap[:] - X_mean) / X_std).astype(np.float32)
    X_tr_n = ((x_tr_mmap[:] - X_mean) / X_std).astype(np.float32)

    Y_cbf_val = y_val_mmap[:, 0].astype(np.float64)
    Y_att_val = y_val_mmap[:, 1].astype(np.float64)

    Y_cbf_tr_std = ((y_tr_mmap[:, 0] - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_tr_std = ((y_tr_mmap[:, 1] - att_mean) / att_std).astype(np.float32)

    val_early_mask = (Y_att_val >= 0.5) & (Y_att_val < 1.0)

    # 3. Extract Penultimate Features to Disk Cache
    f_cbf_tr_path = os.path.join(FEAT_CACHE_DIR, "F_cbf_train.npy")
    f_cbf_val_path = os.path.join(FEAT_CACHE_DIR, "F_cbf_val.npy")
    F_cbf_tr = extract_features_to_mmap(cbf_net, X_tr_n, f_cbf_tr_path)
    F_cbf_val = extract_features_to_mmap(cbf_net, X_val_n, f_cbf_val_path)

    f_att_tr_path = os.path.join(FEAT_CACHE_DIR, "F_att_train.npy")
    f_att_val_path = os.path.join(FEAT_CACHE_DIR, "F_att_val.npy")
    F_att_tr = extract_features_to_mmap(att_net, X_tr_n, f_att_tr_path)
    F_att_val = extract_features_to_mmap(att_net, X_val_n, f_att_val_path)

    # Compute Baseline DNN validation metrics for reference
    pred_cbf_dnn_val = cbf_net.predict_physical(X_val_n)
    pred_att_dnn_val = att_net.predict_physical(X_val_n)
    m_cbf_dnn_val = calc_metrics(Y_cbf_val, pred_cbf_dnn_val)
    m_att_dnn_val = calc_metrics(Y_att_val, pred_att_dnn_val)
    m_early_att_dnn_val = calc_metrics(Y_att_val[val_early_mask], pred_att_dnn_val[val_early_mask])

    print(f"\nBaseline DNN Val Metrics:")
    print(f"  CBF: RMSE = {m_cbf_dnn_val['RMSE']:.4f}, MAE = {m_cbf_dnn_val['MAE']:.4f}, CCC = {m_cbf_dnn_val['CCC']:.4f}")
    print(f"  ATT: RMSE = {m_att_dnn_val['RMSE']:.4f}, MAE = {m_att_dnn_val['MAE']:.4f}, CCC = {m_att_dnn_val['CCC']:.4f}")
    print(f"  Early ATT [0.5, 1.0)s: RMSE = {m_early_att_dnn_val['RMSE']:.4f}, Bias = {m_early_att_dnn_val['Bias']:+.4f}, CCC = {m_early_att_dnn_val['CCC']:.4f}")

    # 4. Search Space Definition
    hidden_counts = [128, 256, 512, 1024, 2048]
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    activations = ["tanh", "elu", "relu"]

    sweep_csv_path = os.path.join(OUT_DIR, "validation_sweep.csv")
    sweep_results = []

    # =========================================================================
    # PART A: CBF HYPERPARAMETER SWEEP
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART A: CBF HYPERPARAMETER SWEEP (VALIDATION SET, N=100,000)")
    print("=" * 70)

    # 1. Sweep Hidden Neurons x Alphas with tanh
    print("Step A1: Sweeping Hidden Counts x Regularization (activation = tanh)...")
    for act in ["tanh"]:
        for nh in hidden_counts:
            t_acc0 = time.time()
            elm = FastGPUChunkedELM(n_hidden=nh, alpha=alphas[0], activation=act, seed=42)
            HtH, HtY = elm.compute_HtH_HtY(F_cbf_tr, Y_cbf_tr_std)
            acc_time = time.time() - t_acc0
            print(f"  [CBF H={nh:4d}, act={act:4s}] HtH accumulated in {acc_time:.2f}s. Evaluating {len(alphas)} alphas...")

            for al in alphas:
                t0 = time.time()
                elm.solve_alpha(HtH, HtY, al)
                pred_std = elm.predict(F_cbf_val)
                pred_phys = np.clip(pred_std * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
                m = calc_metrics(Y_cbf_val, pred_phys)
                el_time = time.time() - t0

                res = {
                    "Target": "CBF", "Stage": "A1_Capacity_Reg",
                    "Hidden_Neurons": nh, "Alpha": al, "Activation": act, "Seed": 42,
                    "Val_RMSE": m["RMSE"], "Val_MAE": m["MAE"], "Val_Bias": m["Bias"],
                    "Val_CCC": m["CCC"], "Val_R2": m["R2"],
                    "Early_ATT_RMSE": np.nan, "Early_ATT_Bias": np.nan, "Early_ATT_CCC": np.nan,
                    "Fit_Time_Sec": el_time + acc_time / len(alphas)
                }
                sweep_results.append(res)
                print(f"    alpha={al:7.1e} -> Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, CCC: {m['CCC']:.4f} ({el_time:.2f}s)")
            del HtH, HtY, elm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pd.DataFrame(sweep_results).to_csv(sweep_csv_path, index=False)

    # Find top 2 hidden counts and top 2 alphas for CBF
    df_cbf_tanh = pd.DataFrame([r for r in sweep_results if r["Target"] == "CBF"])
    top_cbf_cfgs = df_cbf_tanh.sort_values("Val_RMSE").head(3)
    best_cbf_nh = int(top_cbf_cfgs.iloc[0]["Hidden_Neurons"])
    best_cbf_al = float(top_cbf_cfgs.iloc[0]["Alpha"])

    # 2. Sweep Alternative Activations (elu, relu) on best configurations
    print(f"\nStep A2: Evaluating Alternative Activations at promising H={best_cbf_nh} and alpha={best_cbf_al}...")
    for act in ["elu", "relu"]:
        for nh in [best_cbf_nh, 512 if best_cbf_nh != 512 else 1024]:
            elm = FastGPUChunkedELM(n_hidden=nh, alpha=best_cbf_al, activation=act, seed=42)
            HtH, HtY = elm.compute_HtH_HtY(F_cbf_tr, Y_cbf_tr_std)

            for al in [best_cbf_al, 1e-2 if best_cbf_al != 1e-2 else 1e-3]:
                t0 = time.time()
                elm.solve_alpha(HtH, HtY, al)
                pred_std = elm.predict(F_cbf_val)
                pred_phys = np.clip(pred_std * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
                m = calc_metrics(Y_cbf_val, pred_phys)
                el_time = time.time() - t0

                res = {
                    "Target": "CBF", "Stage": "A2_Activation_Test",
                    "Hidden_Neurons": nh, "Alpha": al, "Activation": act, "Seed": 42,
                    "Val_RMSE": m["RMSE"], "Val_MAE": m["MAE"], "Val_Bias": m["Bias"],
                    "Val_CCC": m["CCC"], "Val_R2": m["R2"],
                    "Early_ATT_RMSE": np.nan, "Early_ATT_Bias": np.nan, "Early_ATT_CCC": np.nan,
                    "Fit_Time_Sec": el_time
                }
                sweep_results.append(res)
                print(f"  CBF [H={nh:4d}, alpha={al:7.1e}, act={act:4s}] -> Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, CCC: {m['CCC']:.4f} ({el_time:.2f}s)")
            del HtH, HtY, elm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pd.DataFrame(sweep_results).to_csv(sweep_csv_path, index=False)

    # =========================================================================
    # PART B: ATT HYPERPARAMETER SWEEP
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART B: ATT HYPERPARAMETER SWEEP (VALIDATION SET, N=100,000)")
    print("=" * 70)

    print("Step B1: Sweeping Hidden Counts x Regularization (activation = tanh)...")
    for act in ["tanh"]:
        for nh in hidden_counts:
            t_acc0 = time.time()
            elm = FastGPUChunkedELM(n_hidden=nh, alpha=alphas[0], activation=act, seed=42)
            HtH, HtY = elm.compute_HtH_HtY(F_att_tr, Y_att_tr_std)
            acc_time = time.time() - t_acc0
            print(f"  [ATT H={nh:4d}, act={act:4s}] HtH accumulated in {acc_time:.2f}s. Evaluating {len(alphas)} alphas...")

            for al in alphas:
                t0 = time.time()
                elm.solve_alpha(HtH, HtY, al)
                pred_std = elm.predict(F_att_val)
                pred_phys = np.clip(pred_std * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
                m = calc_metrics(Y_att_val, pred_phys)
                m_early = calc_metrics(Y_att_val[val_early_mask], pred_phys[val_early_mask])
                el_time = time.time() - t0

                res = {
                    "Target": "ATT", "Stage": "B1_Capacity_Reg",
                    "Hidden_Neurons": nh, "Alpha": al, "Activation": act, "Seed": 42,
                    "Val_RMSE": m["RMSE"], "Val_MAE": m["MAE"], "Val_Bias": m["Bias"],
                    "Val_CCC": m["CCC"], "Val_R2": m["R2"],
                    "Early_ATT_RMSE": m_early["RMSE"], "Early_ATT_Bias": m_early["Bias"], "Early_ATT_CCC": m_early["CCC"],
                    "Fit_Time_Sec": el_time + acc_time / len(alphas)
                }
                sweep_results.append(res)
                print(f"    alpha={al:7.1e} -> Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, CCC: {m['CCC']:.4f} | Early RMSE: {m_early['RMSE']:.4f} ({el_time:.2f}s)")
            del HtH, HtY, elm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pd.DataFrame(sweep_results).to_csv(sweep_csv_path, index=False)

    # Find top ATT configurations
    df_att_tanh = pd.DataFrame([r for r in sweep_results if r["Target"] == "ATT"])
    top_att_cfgs = df_att_tanh.sort_values("Val_RMSE").head(3)
    best_att_nh = int(top_att_cfgs.iloc[0]["Hidden_Neurons"])
    best_att_al = float(top_att_cfgs.iloc[0]["Alpha"])

    # 2. Sweep Alternative Activations on best ATT configurations
    print(f"\nStep B2: Evaluating Alternative Activations at promising H={best_att_nh} and alpha={best_att_al}...")
    for act in ["elu", "relu"]:
        for nh in [best_att_nh, 512 if best_att_nh != 512 else 1024]:
            elm = FastGPUChunkedELM(n_hidden=nh, alpha=best_att_al, activation=act, seed=42)
            HtH, HtY = elm.compute_HtH_HtY(F_att_tr, Y_att_tr_std)

            for al in [best_att_al, 1e-2 if best_att_al != 1e-2 else 1e-3]:
                t0 = time.time()
                elm.solve_alpha(HtH, HtY, al)
                pred_std = elm.predict(F_att_val)
                pred_phys = np.clip(pred_std * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
                m = calc_metrics(Y_att_val, pred_phys)
                m_early = calc_metrics(Y_att_val[val_early_mask], pred_phys[val_early_mask])
                el_time = time.time() - t0

                res = {
                    "Target": "ATT", "Stage": "B2_Activation_Test",
                    "Hidden_Neurons": nh, "Alpha": al, "Activation": act, "Seed": 42,
                    "Val_RMSE": m["RMSE"], "Val_MAE": m["MAE"], "Val_Bias": m["Bias"],
                    "Val_CCC": m["CCC"], "Val_R2": m["R2"],
                    "Early_ATT_RMSE": m_early["RMSE"], "Early_ATT_Bias": m_early["Bias"], "Early_ATT_CCC": m_early["CCC"],
                    "Fit_Time_Sec": el_time
                }
                sweep_results.append(res)
                print(f"  ATT [H={nh:4d}, alpha={al:7.1e}, act={act:4s}] -> Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, CCC: {m['CCC']:.4f} | Early RMSE: {m_early['RMSE']:.4f} ({el_time:.2f}s)")
            del HtH, HtY, elm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pd.DataFrame(sweep_results).to_csv(sweep_csv_path, index=False)

    df_full_sweep = pd.DataFrame(sweep_results)
    df_full_sweep.to_csv(sweep_csv_path, index=False)
    print(f"\nSaved full hyperparameter sweep ({len(df_full_sweep)} configurations) to: {sweep_csv_path}")

    # =========================================================================
    # PART C: SEED VARIABILITY & ENSEMBLE EVALUATION (10 SEEDS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART C: SEED VARIABILITY ANALYSIS (10 RANDOM SEEDS: 42-51)")
    print("=" * 70)

    # Identify the top 2 overall candidate configurations for CBF and ATT
    df_cbf_all = df_full_sweep[df_full_sweep["Target"] == "CBF"].sort_values("Val_RMSE")
    df_att_all = df_full_sweep[df_full_sweep["Target"] == "ATT"].sort_values("Val_RMSE")

    print("\nTop 3 CBF Configurations from Validation Sweep:")
    print(df_cbf_all.head(3)[["Hidden_Neurons", "Alpha", "Activation", "Val_RMSE", "Val_MAE", "Val_CCC"]].to_string(index=False))

    print("\nTop 3 ATT Configurations from Validation Sweep:")
    print(df_att_all.head(3)[["Hidden_Neurons", "Alpha", "Activation", "Val_RMSE", "Val_MAE", "Early_ATT_RMSE", "Val_CCC"]].to_string(index=False))

    seeds = list(range(42, 52)) # 10 seeds: 42..51
    seed_records = []

    # 1. Top CBF Candidate Seed Sweep
    cbf_cand_cfg = {
        "nh": int(df_cbf_all.iloc[0]["Hidden_Neurons"]),
        "al": float(df_cbf_all.iloc[0]["Alpha"]),
        "act": str(df_cbf_all.iloc[0]["Activation"])
    }
    # Also evaluate the current baseline config (512, 1e-2, tanh) across the same 10 seeds
    cbf_baseline_cfg = {"nh": 512, "al": 1e-2, "act": "tanh"}

    for cfg_label, cfg in [("Tuned_CBF_Candidate", cbf_cand_cfg), ("Current_CBF_Baseline", cbf_baseline_cfg)]:
        cbf_preds_all = []
        for s in seeds:
            elm = FastGPUChunkedELM(n_hidden=cfg["nh"], alpha=cfg["al"], activation=cfg["act"], seed=s)
            elm.fit_chunked(F_cbf_tr, Y_cbf_tr_std)
            p_val = np.clip(elm.predict(F_cbf_val) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
            cbf_preds_all.append(p_val)
            m = calc_metrics(Y_cbf_val, p_val)
            seed_records.append({
                "Target": "CBF", "Configuration_Name": cfg_label,
                "Hidden_Neurons": cfg["nh"], "Alpha": cfg["al"], "Activation": cfg["act"], "Seed": s,
                "Val_RMSE": m["RMSE"], "Val_MAE": m["MAE"], "Val_CCC": m["CCC"], "Val_Bias": m["Bias"],
                "Early_ATT_RMSE": np.nan, "Early_ATT_Bias": np.nan, "Early_ATT_CCC": np.nan
            })
            del elm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate 5-seed ensemble and 10-seed ensemble
        ens_5 = np.mean(cbf_preds_all[:5], axis=0)
        ens_10 = np.mean(cbf_preds_all, axis=0)
        m_ens5 = calc_metrics(Y_cbf_val, ens_5)
        m_ens10 = calc_metrics(Y_cbf_val, ens_10)

        seed_records.append({
            "Target": "CBF", "Configuration_Name": f"{cfg_label}_5Seed_Ensemble",
            "Hidden_Neurons": cfg["nh"], "Alpha": cfg["al"], "Activation": cfg["act"], "Seed": "Ensemble_5",
            "Val_RMSE": m_ens5["RMSE"], "Val_MAE": m_ens5["MAE"], "Val_CCC": m_ens5["CCC"], "Val_Bias": m_ens5["Bias"],
            "Early_ATT_RMSE": np.nan, "Early_ATT_Bias": np.nan, "Early_ATT_CCC": np.nan
        })
        seed_records.append({
            "Target": "CBF", "Configuration_Name": f"{cfg_label}_10Seed_Ensemble",
            "Hidden_Neurons": cfg["nh"], "Alpha": cfg["al"], "Activation": cfg["act"], "Seed": "Ensemble_10",
            "Val_RMSE": m_ens10["RMSE"], "Val_MAE": m_ens10["MAE"], "Val_CCC": m_ens10["CCC"], "Val_Bias": m_ens10["Bias"],
            "Early_ATT_RMSE": np.nan, "Early_ATT_Bias": np.nan, "Early_ATT_CCC": np.nan
        })

    # 2. Top ATT Candidate Seed Sweep
    att_cand_cfg = {
        "nh": int(df_att_all.iloc[0]["Hidden_Neurons"]),
        "al": float(df_att_all.iloc[0]["Alpha"]),
        "act": str(df_att_all.iloc[0]["Activation"])
    }
    att_baseline_cfg = {"nh": 512, "al": 1e-2, "act": "tanh"}

    for cfg_label, cfg in [("Tuned_ATT_Candidate", att_cand_cfg), ("Current_ATT_Baseline", att_baseline_cfg)]:
        att_preds_all = []
        for s in seeds:
            elm = FastGPUChunkedELM(n_hidden=cfg["nh"], alpha=cfg["al"], activation=cfg["act"], seed=s)
            elm.fit_chunked(F_att_tr, Y_att_tr_std)
            p_val = np.clip(elm.predict(F_att_val) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
            att_preds_all.append(p_val)
            m = calc_metrics(Y_att_val, p_val)
            m_early = calc_metrics(Y_att_val[val_early_mask], p_val[val_early_mask])
            seed_records.append({
                "Target": "ATT", "Configuration_Name": cfg_label,
                "Hidden_Neurons": cfg["nh"], "Alpha": cfg["al"], "Activation": cfg["act"], "Seed": s,
                "Val_RMSE": m["RMSE"], "Val_MAE": m["MAE"], "Val_CCC": m["CCC"], "Val_Bias": m["Bias"],
                "Early_ATT_RMSE": m_early["RMSE"], "Early_ATT_Bias": m_early["Bias"], "Early_ATT_CCC": m_early["CCC"]
            })
            del elm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate 5-seed ensemble and 10-seed ensemble
        ens_5 = np.mean(att_preds_all[:5], axis=0)
        ens_10 = np.mean(att_preds_all, axis=0)
        m_ens5 = calc_metrics(Y_att_val, ens_5)
        m_ens10 = calc_metrics(Y_att_val, ens_10)
        m_early5 = calc_metrics(Y_att_val[val_early_mask], ens_5[val_early_mask])
        m_early10 = calc_metrics(Y_att_val[val_early_mask], ens_10[val_early_mask])

        seed_records.append({
            "Target": "ATT", "Configuration_Name": f"{cfg_label}_5Seed_Ensemble",
            "Hidden_Neurons": cfg["nh"], "Alpha": cfg["al"], "Activation": cfg["act"], "Seed": "Ensemble_5",
            "Val_RMSE": m_ens5["RMSE"], "Val_MAE": m_ens5["MAE"], "Val_CCC": m_ens5["CCC"], "Val_Bias": m_ens5["Bias"],
            "Early_ATT_RMSE": m_early5["RMSE"], "Early_ATT_Bias": m_early5["Bias"], "Early_ATT_CCC": m_early5["CCC"]
        })
        seed_records.append({
            "Target": "ATT", "Configuration_Name": f"{cfg_label}_10Seed_Ensemble",
            "Hidden_Neurons": cfg["nh"], "Alpha": cfg["al"], "Activation": cfg["act"], "Seed": "Ensemble_10",
            "Val_RMSE": m_ens10["RMSE"], "Val_MAE": m_ens10["MAE"], "Val_CCC": m_ens10["CCC"], "Val_Bias": m_ens10["Bias"],
            "Early_ATT_RMSE": m_early10["RMSE"], "Early_ATT_Bias": m_early10["Bias"], "Early_ATT_CCC": m_early10["CCC"]
        })

    df_seeds = pd.DataFrame(seed_records)
    seed_csv_path = os.path.join(OUT_DIR, "seed_variability.csv")
    df_seeds.to_csv(seed_csv_path, index=False)
    print(f"\nSaved seed variability analysis to: {seed_csv_path}")

    # =========================================================================
    # PART D: VALIDATION COMPARISON SUMMARY
    # =========================================================================
    comp_rows = [
        {
            "Model": "Baseline A (DNN-G)", "Target": "CBF", "Configuration": "StandardizedNet (Frozen)",
            "Val_RMSE": m_cbf_dnn_val["RMSE"], "Val_MAE": m_cbf_dnn_val["MAE"], "Val_CCC": m_cbf_dnn_val["CCC"],
            "Delta_Val_RMSE": 0.0, "Gain_Val_RMSE_%": 0.0
        },
        {
            "Model": "Current Baseline Feature ELM (5 seeds)", "Target": "CBF", "Configuration": "H=512, alpha=1e-2, tanh (5 seeds)",
            "Val_RMSE": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Current_CBF_Baseline_5Seed_Ensemble")]["Val_RMSE"].values[0],
            "Val_MAE": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Current_CBF_Baseline_5Seed_Ensemble")]["Val_MAE"].values[0],
            "Val_CCC": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Current_CBF_Baseline_5Seed_Ensemble")]["Val_CCC"].values[0],
            "Delta_Val_RMSE": m_cbf_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Current_CBF_Baseline_5Seed_Ensemble")]["Val_RMSE"].values[0],
            "Gain_Val_RMSE_%": 100.0 * (m_cbf_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Current_CBF_Baseline_5Seed_Ensemble")]["Val_RMSE"].values[0]) / m_cbf_dnn_val["RMSE"]
        },
        {
            "Model": "Tuned ELM Single Best", "Target": "CBF",
            "Configuration": f"H={cbf_cand_cfg['nh']}, alpha={cbf_cand_cfg['al']}, {cbf_cand_cfg['act']} (seed 42)",
            "Val_RMSE": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate") & (df_seeds["Seed"]==42)]["Val_RMSE"].values[0],
            "Val_MAE": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate") & (df_seeds["Seed"]==42)]["Val_MAE"].values[0],
            "Val_CCC": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate") & (df_seeds["Seed"]==42)]["Val_CCC"].values[0],
            "Delta_Val_RMSE": m_cbf_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate") & (df_seeds["Seed"]==42)]["Val_RMSE"].values[0],
            "Gain_Val_RMSE_%": 100.0 * (m_cbf_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate") & (df_seeds["Seed"]==42)]["Val_RMSE"].values[0]) / m_cbf_dnn_val["RMSE"]
        },
        {
            "Model": "Tuned ELM Ensemble (5 seeds)", "Target": "CBF",
            "Configuration": f"H={cbf_cand_cfg['nh']}, alpha={cbf_cand_cfg['al']}, {cbf_cand_cfg['act']} (5 seeds)",
            "Val_RMSE": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_5Seed_Ensemble")]["Val_RMSE"].values[0],
            "Val_MAE": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_5Seed_Ensemble")]["Val_MAE"].values[0],
            "Val_CCC": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_5Seed_Ensemble")]["Val_CCC"].values[0],
            "Delta_Val_RMSE": m_cbf_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_5Seed_Ensemble")]["Val_RMSE"].values[0],
            "Gain_Val_RMSE_%": 100.0 * (m_cbf_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_5Seed_Ensemble")]["Val_RMSE"].values[0]) / m_cbf_dnn_val["RMSE"]
        },
        {
            "Model": "Tuned ELM Ensemble (10 seeds)", "Target": "CBF",
            "Configuration": f"H={cbf_cand_cfg['nh']}, alpha={cbf_cand_cfg['al']}, {cbf_cand_cfg['act']} (10 seeds)",
            "Val_RMSE": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_10Seed_Ensemble")]["Val_RMSE"].values[0],
            "Val_MAE": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_10Seed_Ensemble")]["Val_MAE"].values[0],
            "Val_CCC": df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_10Seed_Ensemble")]["Val_CCC"].values[0],
            "Delta_Val_RMSE": m_cbf_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_10Seed_Ensemble")]["Val_RMSE"].values[0],
            "Gain_Val_RMSE_%": 100.0 * (m_cbf_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="CBF") & (df_seeds["Configuration_Name"]=="Tuned_CBF_Candidate_10Seed_Ensemble")]["Val_RMSE"].values[0]) / m_cbf_dnn_val["RMSE"]
        },

        # ATT
        {
            "Model": "Baseline A (DNN-G)", "Target": "ATT", "Configuration": "StandardizedNet (Frozen)",
            "Val_RMSE": m_att_dnn_val["RMSE"], "Val_MAE": m_att_dnn_val["MAE"], "Val_CCC": m_att_dnn_val["CCC"],
            "Delta_Val_RMSE": 0.0, "Gain_Val_RMSE_%": 0.0
        },
        {
            "Model": "Current Baseline Feature ELM (5 seeds)", "Target": "ATT", "Configuration": "H=512, alpha=1e-2, tanh (5 seeds)",
            "Val_RMSE": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Current_ATT_Baseline_5Seed_Ensemble")]["Val_RMSE"].values[0],
            "Val_MAE": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Current_ATT_Baseline_5Seed_Ensemble")]["Val_MAE"].values[0],
            "Val_CCC": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Current_ATT_Baseline_5Seed_Ensemble")]["Val_CCC"].values[0],
            "Delta_Val_RMSE": m_att_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Current_ATT_Baseline_5Seed_Ensemble")]["Val_RMSE"].values[0],
            "Gain_Val_RMSE_%": 100.0 * (m_att_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Current_ATT_Baseline_5Seed_Ensemble")]["Val_RMSE"].values[0]) / m_att_dnn_val["RMSE"]
        },
        {
            "Model": "Tuned ELM Single Best", "Target": "ATT",
            "Configuration": f"H={att_cand_cfg['nh']}, alpha={att_cand_cfg['al']}, {att_cand_cfg['act']} (seed 42)",
            "Val_RMSE": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate") & (df_seeds["Seed"]==42)]["Val_RMSE"].values[0],
            "Val_MAE": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate") & (df_seeds["Seed"]==42)]["Val_MAE"].values[0],
            "Val_CCC": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate") & (df_seeds["Seed"]==42)]["Val_CCC"].values[0],
            "Delta_Val_RMSE": m_att_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate") & (df_seeds["Seed"]==42)]["Val_RMSE"].values[0],
            "Gain_Val_RMSE_%": 100.0 * (m_att_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate") & (df_seeds["Seed"]==42)]["Val_RMSE"].values[0]) / m_att_dnn_val["RMSE"]
        },
        {
            "Model": "Tuned ELM Ensemble (5 seeds)", "Target": "ATT",
            "Configuration": f"H={att_cand_cfg['nh']}, alpha={att_cand_cfg['al']}, {att_cand_cfg['act']} (5 seeds)",
            "Val_RMSE": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_5Seed_Ensemble")]["Val_RMSE"].values[0],
            "Val_MAE": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_5Seed_Ensemble")]["Val_MAE"].values[0],
            "Val_CCC": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_5Seed_Ensemble")]["Val_CCC"].values[0],
            "Delta_Val_RMSE": m_att_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_5Seed_Ensemble")]["Val_RMSE"].values[0],
            "Gain_Val_RMSE_%": 100.0 * (m_att_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_5Seed_Ensemble")]["Val_RMSE"].values[0]) / m_att_dnn_val["RMSE"]
        },
        {
            "Model": "Tuned ELM Ensemble (10 seeds)", "Target": "ATT",
            "Configuration": f"H={att_cand_cfg['nh']}, alpha={att_cand_cfg['al']}, {att_cand_cfg['act']} (10 seeds)",
            "Val_RMSE": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_10Seed_Ensemble")]["Val_RMSE"].values[0],
            "Val_MAE": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_10Seed_Ensemble")]["Val_MAE"].values[0],
            "Val_CCC": df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_10Seed_Ensemble")]["Val_CCC"].values[0],
            "Delta_Val_RMSE": m_att_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_10Seed_Ensemble")]["Val_RMSE"].values[0],
            "Gain_Val_RMSE_%": 100.0 * (m_att_dnn_val["RMSE"] - df_seeds[(df_seeds["Target"]=="ATT") & (df_seeds["Configuration_Name"]=="Tuned_ATT_Candidate_10Seed_Ensemble")]["Val_RMSE"].values[0]) / m_att_dnn_val["RMSE"]
        }
    ]

    df_comp = pd.DataFrame(comp_rows)
    val_comp_path = os.path.join(OUT_DIR, "validation_comparison.csv")
    df_comp.to_csv(val_comp_path, index=False)
    print(f"\nSaved validation comparison summary to: {val_comp_path}")
    print(df_comp[["Model", "Target", "Configuration", "Val_RMSE", "Gain_Val_RMSE_%", "Val_MAE", "Val_CCC"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("STAGE 2 & 3 COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()

