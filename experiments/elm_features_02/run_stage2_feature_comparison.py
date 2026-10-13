"""
run_stage2_feature_comparison.py
Stage 2: Systematic comparison of feature representations for ELM readout.
Evaluates 4 input representations on validation cohort (N=100,000):
  A. Existing DNN penultimate-layer features (CBF: 128-dim, ATT: 64-dim)
  B. DNN features + 4 Raw ASL measurements (CBF: 132-dim, ATT: 68-dim)
  C. DNN features + Normalized signals and non-linear contrast differences (CBF: 135-dim, ATT: 71-dim)
  D. DNN features + Physically derived ASL features (CBF: 138-dim, ATT: 74-dim)
Evaluates feature conditioning, checks for rank deficiency, and tunes ridge regularization
exclusively on validation data.
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

class FastGPUChunkedELM(GPUChunkedELM):
    """ELM supporting single H^T H accumulation and multiple alpha regularizations."""
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

def main():
    print("=" * 80)
    print("STAGE 2: FEATURE REPRESENTATIONS COMPARISON (LEAKAGE-FREE VALIDATION)")
    print(f"Compute Device: {device}")
    print("=" * 80)

    # 1. Normalization parameters & Baseline Models
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
    # Training sample: 100k representative training set (seed 2026) for representation comparison
    N_tr = 100_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    X_tr_raw = x_tr_mmap[idx_tr]
    Y_cbf_tr = y_tr_mmap[idx_tr, 0]
    Y_att_tr = y_tr_mmap[idx_tr, 1]
    Y_cbf_tr_std = ((Y_cbf_tr - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_tr_std = ((Y_att_tr - att_mean) / att_std).astype(np.float32)

    X_val_raw = x_val_mmap[:]
    X_val_n = ((X_val_raw - X_mean) / X_std).astype(np.float32)
    Y_cbf_val = y_val_mmap[:, 0].astype(np.float64)
    Y_att_val = y_val_mmap[:, 1].astype(np.float64)

    val_early_mask = (Y_att_val >= 0.5) & (Y_att_val < 1.0)

    # 3. Compute Control Metrics: Baseline A DNN
    pred_cbf_dnn_val = cbf_net.predict_physical(X_val_n)
    pred_att_dnn_val = att_net.predict_physical(X_val_n)
    m_cbf_dnn = calc_metrics(Y_cbf_val, pred_cbf_dnn_val)
    m_att_dnn = calc_metrics(Y_att_val, pred_att_dnn_val)
    m_early_dnn = calc_metrics(Y_att_val[val_early_mask], pred_att_dnn_val[val_early_mask])

    print(f"\nControl Baseline A DNN Val Metrics:")
    print(f"  CBF: RMSE = {m_cbf_dnn['RMSE']:.4f}, MAE = {m_cbf_dnn['MAE']:.4f}, CCC = {m_cbf_dnn['CCC']:.4f}")
    print(f"  ATT: RMSE = {m_att_dnn['RMSE']:.4f}, MAE = {m_att_dnn['MAE']:.4f}, CCC = {m_att_dnn['CCC']:.4f}")
    print(f"  Early ATT: RMSE = {m_early_dnn['RMSE']:.4f}, Bias = {m_early_dnn['Bias']:+.4f}")

    # 4. Extract Penultimate Latents (Representation A)
    print("\nExtracting Penultimate Latents from Frozen Backbones...")
    X_tr_n = ((X_tr_raw - X_mean) / X_std).astype(np.float32)

    with torch.no_grad():
        F_lat_cbf_tr = cbf_net.penultimate_features(torch.from_numpy(X_tr_n).to(device))
        F_lat_cbf_val = cbf_net.penultimate_features(torch.from_numpy(X_val_n).to(device))
        F_lat_att_tr = att_net.penultimate_features(torch.from_numpy(X_tr_n).to(device))
        F_lat_att_val = att_net.penultimate_features(torch.from_numpy(X_val_n).to(device))

    # 5. Extract and Scale Auxiliary Features
    print("Generating Signal-Derived Feature Representations...")
    # Rep B: Raw Signals
    F_raw_tr = fe.extract_raw_features(X_tr_raw)
    F_raw_val = fe.extract_raw_features(X_val_raw)

    # Rep C: Normalized Signals + Relative Contrasts
    F_normdiff_tr = fe.extract_normalized_diff_features(X_tr_raw, X_mean, X_std)
    F_normdiff_val = fe.extract_normalized_diff_features(X_val_raw, X_mean, X_std)

    # Rep D: Physical Features
    F_phys_tr_raw = fe.extract_physics_features(X_tr_raw)
    F_phys_val_raw = fe.extract_physics_features(X_val_raw)
    mu_phys, std_phys = fe.fit_feature_scaler(F_phys_tr_raw)
    F_phys_tr = fe.apply_feature_scaler(F_phys_tr_raw, mu_phys, std_phys)
    F_phys_val = fe.apply_feature_scaler(F_phys_val_raw, mu_phys, std_phys)

    # 6. Audit Feature Conditioning & Numerical Stability
    cond_records = [
        fe.check_feature_conditioning(F_lat_cbf_tr, "Rep_A_Latents_CBF"),
        fe.check_feature_conditioning(F_lat_att_tr, "Rep_A_Latents_ATT"),
        fe.check_feature_conditioning(F_raw_tr, "Rep_B_Aux_Raw"),
        fe.check_feature_conditioning(F_normdiff_tr, "Rep_C_Aux_NormDiff"),
        fe.check_feature_conditioning(F_phys_tr, "Rep_D_Aux_Physics")
    ]
    df_cond = pd.DataFrame(cond_records)
    cond_csv = os.path.join(OUT_DIR, "feature_conditioning_audit.csv")
    df_cond.to_csv(cond_csv, index=False)
    print(f"\nSaved Feature Conditioning Audit to: {cond_csv}")
    print(df_cond.to_string(index=False))

    # 7. Construct Full Feature Sets for CBF and ATT
    feature_sets = {
        "Rep_A_Latents_Only": {
            "CBF_tr": F_lat_cbf_tr, "CBF_val": F_lat_cbf_val,
            "ATT_tr": F_lat_att_tr, "ATT_val": F_lat_att_val,
            "desc": "Penultimate Latents Only (CBF: 128, ATT: 64)"
        },
        "Rep_B_Latents_Plus_Raw": {
            "CBF_tr": np.hstack([F_lat_cbf_tr, F_raw_tr]),
            "CBF_val": np.hstack([F_lat_cbf_val, F_raw_val]),
            "ATT_tr": np.hstack([F_lat_att_tr, F_raw_tr]),
            "ATT_val": np.hstack([F_lat_att_val, F_raw_val]),
            "desc": "Latents + 4 Raw ASL measurements"
        },
        "Rep_C_Latents_Plus_NormDiff": {
            "CBF_tr": np.hstack([F_lat_cbf_tr, F_normdiff_tr]),
            "CBF_val": np.hstack([F_lat_cbf_val, F_normdiff_val]),
            "ATT_tr": np.hstack([F_lat_att_tr, F_normdiff_tr]),
            "ATT_val": np.hstack([F_lat_att_val, F_normdiff_val]),
            "desc": "Latents + 4 Normalized signals + 3 Relative contrasts"
        },
        "Rep_D_Latents_Plus_Physics": {
            "CBF_tr": np.hstack([F_lat_cbf_tr, F_phys_tr]),
            "CBF_val": np.hstack([F_lat_cbf_val, F_phys_val]),
            "ATT_tr": np.hstack([F_lat_att_tr, F_phys_tr]),
            "ATT_val": np.hstack([F_lat_att_val, F_phys_val]),
            "desc": "Latents + 10 Scaled Physics features (slopes, ratios, AUC, decay)"
        }
    }

    # 8. Regularization Sweep for each Feature Representation on Validation Set
    alphas_to_sweep = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    sweep_records = []

    print("\n" + "=" * 70)
    print("SWEEPING REGULARIZATION ACROSS FEATURE REPRESENTATIONS")
    print("=" * 70)

    for rep_name, rdata in feature_sets.items():
        print(f"\nEvaluating Representation: {rep_name} ({rdata['desc']})")
        # --- CBF Optimization ---
        # Fixed capacity H=2048, activation=elu (optimal from Exp 1)
        elm_cbf = FastGPUChunkedELM(n_hidden=2048, alpha=1.0, activation="elu", seed=42)
        HtH_cbf, HtY_cbf = elm_cbf.compute_HtH_HtY(rdata["CBF_tr"], Y_cbf_tr_std)

        best_cbf_rmse = float('inf')
        best_cbf_alpha = None
        best_cbf_metrics = None

        for al in alphas_to_sweep:
            elm_cbf.solve_alpha(HtH_cbf, HtY_cbf, al)
            p_val = np.clip(elm_cbf.predict(rdata["CBF_val"]) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
            m = calc_metrics(Y_cbf_val, p_val)
            if m["RMSE"] < best_cbf_rmse:
                best_cbf_rmse = m["RMSE"]
                best_cbf_alpha = al
                best_cbf_metrics = m
            sweep_records.append({
                "Target": "CBF", "Representation": rep_name, "Alpha": al,
                "RMSE": m["RMSE"], "MAE": m["MAE"], "CCC": m["CCC"],
                "Early_ATT_RMSE": np.nan, "Early_ATT_Bias": np.nan, "Early_ATT_CCC": np.nan
            })

        print(f"  CBF Best: alpha={best_cbf_alpha:.1e} -> RMSE={best_cbf_metrics['RMSE']:.4f}, MAE={best_cbf_metrics['MAE']:.4f}, CCC={best_cbf_metrics['CCC']:.4f}")
        del HtH_cbf, HtY_cbf, elm_cbf

        # --- ATT Optimization ---
        # Fixed capacity H=2048, activation=tanh (optimal from Exp 1)
        elm_att = FastGPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=42)
        HtH_att, HtY_att = elm_att.compute_HtH_HtY(rdata["ATT_tr"], Y_att_tr_std)

        best_att_rmse = float('inf')
        best_att_alpha = None
        best_att_metrics = None
        best_att_early = None

        for al in alphas_to_sweep:
            elm_att.solve_alpha(HtH_att, HtY_att, al)
            p_val = np.clip(elm_att.predict(rdata["ATT_val"]) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
            m = calc_metrics(Y_att_val, p_val)
            m_early = calc_metrics(Y_att_val[val_early_mask], p_val[val_early_mask])
            if m["RMSE"] < best_att_rmse:
                best_att_rmse = m["RMSE"]
                best_att_alpha = al
                best_att_metrics = m
                best_att_early = m_early
            sweep_records.append({
                "Target": "ATT", "Representation": rep_name, "Alpha": al,
                "RMSE": m["RMSE"], "MAE": m["MAE"], "CCC": m["CCC"],
                "Early_ATT_RMSE": m_early["RMSE"], "Early_ATT_Bias": m_early["Bias"], "Early_ATT_CCC": m_early["CCC"]
            })

        print(f"  ATT Best: alpha={best_att_alpha:.1e} -> RMSE={best_att_metrics['RMSE']:.4f}, MAE={best_att_metrics['MAE']:.4f}, CCC={best_att_metrics['CCC']:.4f} | Early RMSE={best_att_early['RMSE']:.4f}")
        del HtH_att, HtY_att, elm_att
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_sweep = pd.DataFrame(sweep_records)
    sweep_csv = os.path.join(OUT_DIR, "stage2_feature_sweep.csv")
    df_sweep.to_csv(sweep_csv, index=False)
    print(f"\nSaved full Stage 2 regularization sweep to: {sweep_csv}")

    # 9. Summary Comparison Table (Top Config per Representation)
    summary_rows = [
        {
            "Representation": "Baseline A (DNN-G)", "Target": "CBF", "Optimal_Alpha": "N/A",
            "Val_RMSE": m_cbf_dnn["RMSE"], "Val_MAE": m_cbf_dnn["MAE"], "Val_CCC": m_cbf_dnn["CCC"],
            "Gain_vs_DNN_%": 0.0, "Early_ATT_RMSE": np.nan
        },
        {
            "Representation": "Baseline A (DNN-G)", "Target": "ATT", "Optimal_Alpha": "N/A",
            "Val_RMSE": m_att_dnn["RMSE"], "Val_MAE": m_att_dnn["MAE"], "Val_CCC": m_att_dnn["CCC"],
            "Gain_vs_DNN_%": 0.0, "Early_ATT_RMSE": m_early_dnn["RMSE"]
        }
    ]

    for rep_name in feature_sets.keys():
        sub_cbf = df_sweep[(df_sweep["Target"] == "CBF") & (df_sweep["Representation"] == rep_name)].sort_values("RMSE").iloc[0]
        sub_att = df_sweep[(df_sweep["Target"] == "ATT") & (df_sweep["Representation"] == rep_name)].sort_values("RMSE").iloc[0]

        summary_rows.append({
            "Representation": rep_name, "Target": "CBF", "Optimal_Alpha": f"{sub_cbf['Alpha']:.1e}",
            "Val_RMSE": sub_cbf["RMSE"], "Val_MAE": sub_cbf["MAE"], "Val_CCC": sub_cbf["CCC"],
            "Gain_vs_DNN_%": 100.0 * (m_cbf_dnn["RMSE"] - sub_cbf["RMSE"]) / m_cbf_dnn["RMSE"],
            "Early_ATT_RMSE": np.nan
        })
        summary_rows.append({
            "Representation": rep_name, "Target": "ATT", "Optimal_Alpha": f"{sub_att['Alpha']:.1e}",
            "Val_RMSE": sub_att["RMSE"], "Val_MAE": sub_att["MAE"], "Val_CCC": sub_att["CCC"],
            "Gain_vs_DNN_%": 100.0 * (m_att_dnn["RMSE"] - sub_att["RMSE"]) / m_att_dnn["RMSE"],
            "Early_ATT_RMSE": sub_att["Early_ATT_RMSE"]
        })

    df_summary = pd.DataFrame(summary_rows)
    comp_csv = os.path.join(OUT_DIR, "stage2_feature_comparison.csv")
    df_summary.to_csv(comp_csv, index=False)
    print(f"\nSaved Stage 2 Feature Comparison Summary to: {comp_csv}")
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("STAGE 2 FEATURE COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()

