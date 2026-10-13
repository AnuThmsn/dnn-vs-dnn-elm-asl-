"""
run_stage3_residual_elm.py
Stage 3: Train and evaluate Leakage-Free DNN Residual ELM.
Methodology:
- Constructs true Out-Of-Fold (OOF) DNN predictions on a 100k training partition using
  5-fold cross-validation (seed 2026).
- Calculates leakage-free residual targets:
    CBF residual = true CBF minus OOF-predicted CBF
    ATT residual = true ATT minus OOF-predicted ATT
- Compares four models:
    1. Existing Feature ELM (Latents only -> direct parameter target)
    2. Feature-Fusion ELM (Latents + permitted signal features -> direct parameter target)
    3. Out-Of-Fold Residual ELM (Latents only -> residual target)
    4. Feature-Fusion Residual ELM (Latents + permitted signal features -> residual target)
- At inference on Validation data:
    Generates Baseline A prediction, predicts ELM correction, applies justified physical clipping.
- Tests damping factors lambda in [0.1, 0.25, 0.5, 1.0] and 5-seed ensembles (42..46).
- Saves:
    stage3_residual_comparison.csv
    oof_residual_statistics.csv
    oof_predictions_train.npz
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

def generate_or_load_oof_predictions(X_tr_n, Y_cbf_tr, Y_att_tr, cbf_mean, cbf_std, att_mean, att_std, oof_cache_path):
    """
    Generate 5-fold CV out-of-fold predictions or load existing cache.
    Eliminates in-sample memorization so residual ELM learns genuine generalization error.
    """
    if os.path.exists(oof_cache_path):
        print(f"Loading cached OOF predictions from {oof_cache_path}...")
        data = np.load(oof_cache_path)
        return data["oof_cbf_pred"], data["oof_att_pred"]

    print("Generating 5-Fold Cross-Validated Out-Of-Fold Predictions (100k samples)...")
    N = len(X_tr_n)
    K = 5
    fold_size = N // K
    rng = np.random.default_rng(2026)
    perm = rng.permutation(N)

    oof_cbf_pred = np.empty(N, dtype=np.float32)
    oof_att_pred = np.empty(N, dtype=np.float32)

    Y_cbf_tr_std = ((Y_cbf_tr - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_tr_std = ((Y_att_tr - att_mean) / att_std).astype(np.float32)

    criterion_mse = nn.MSELoss()
    b_sz = 1024

    for k in range(K):
        t0 = time.time()
        val_idx = perm[k * fold_size : (k + 1) * fold_size]
        val_mask = np.zeros(N, dtype=bool)
        val_mask[val_idx] = True
        tr_mask = ~val_mask

        # Train fold CBFnet
        net_c = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
        opt_c = torch.optim.Adam(net_c.parameters(), lr=1e-3)

        # Train fold ATTnet
        net_a = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
        opt_a = torch.optim.Adam(net_a.parameters(), lr=1e-3)

        x_tr_fold = torch.from_numpy(X_tr_n[tr_mask]).to(device)
        yc_tr_fold = torch.from_numpy(Y_cbf_tr_std[tr_mask]).unsqueeze(1).to(device)
        ya_tr_fold = torch.from_numpy(Y_att_tr_std[tr_mask]).unsqueeze(1).to(device)

        n_batches = int(np.ceil(np.sum(tr_mask) / b_sz))
        for ep in range(12):
            for b in range(n_batches):
                bx = x_tr_fold[b * b_sz : (b + 1) * b_sz]
                by_c = yc_tr_fold[b * b_sz : (b + 1) * b_sz]
                by_a = ya_tr_fold[b * b_sz : (b + 1) * b_sz]

                opt_c.zero_grad()
                loss_c = criterion_mse(net_c(bx), by_c)
                loss_c.backward()
                opt_c.step()

                opt_a.zero_grad()
                loss_a = criterion_mse(net_a(bx), by_a)
                loss_a.backward()
                opt_a.step()

        net_c.eval()
        net_a.eval()
        with torch.no_grad():
            oof_cbf_pred[val_mask] = net_c.predict_physical(X_tr_n[val_mask])
            oof_att_pred[val_mask] = net_a.predict_physical(X_tr_n[val_mask])

        el_time = time.time() - t0
        print(f"  Fold {k + 1}/{K} completed in {el_time:.2f}s.")
        del net_c, net_a, opt_c, opt_a, x_tr_fold, yc_tr_fold, ya_tr_fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez_compressed(
        oof_cache_path,
        oof_cbf_pred=oof_cbf_pred,
        oof_att_pred=oof_att_pred
    )
    print(f"Saved OOF predictions cache to {oof_cache_path}.")
    return oof_cbf_pred, oof_att_pred

def main():
    print("=" * 80)
    print("STAGE 3: LEAKAGE-FREE DNN RESIDUAL ELM EVALUATION")
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
    N_tr = 100_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    X_tr_raw = x_tr_mmap[idx_tr]
    Y_cbf_tr = y_tr_mmap[idx_tr, 0].astype(np.float32)
    Y_att_tr = y_tr_mmap[idx_tr, 1].astype(np.float32)
    Y_cbf_tr_std = ((Y_cbf_tr - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_tr_std = ((Y_att_tr - att_mean) / att_std).astype(np.float32)

    X_tr_n = ((X_tr_raw - X_mean) / X_std).astype(np.float32)

    X_val_raw = x_val_mmap[:]
    X_val_n = ((X_val_raw - X_mean) / X_std).astype(np.float32)
    Y_cbf_val = y_val_mmap[:, 0].astype(np.float64)
    Y_att_val = y_val_mmap[:, 1].astype(np.float64)

    val_early_mask = (Y_att_val >= 0.5) & (Y_att_val < 1.0)

    # 3. Baseline A Control Predictions
    p_cbf_dnn_val = cbf_net.predict_physical(X_val_n).astype(np.float64)
    p_att_dnn_val = att_net.predict_physical(X_val_n).astype(np.float64)
    m_base_c = calc_metrics(Y_cbf_val, p_cbf_dnn_val)
    m_base_a = calc_metrics(Y_att_val, p_att_dnn_val)
    m_base_early = calc_metrics(Y_att_val[val_early_mask], p_att_dnn_val[val_early_mask])

    print(f"\nBaseline A DNN Control Metrics:")
    print(f"  CBF: RMSE = {m_base_c['RMSE']:.4f}, MAE = {m_base_c['MAE']:.4f}, CCC = {m_base_c['CCC']:.4f}")
    print(f"  ATT: RMSE = {m_base_a['RMSE']:.4f}, MAE = {m_base_a['MAE']:.4f}, CCC = {m_base_a['CCC']:.4f}")
    print(f"  Early ATT: RMSE = {m_base_early['RMSE']:.4f}, Bias = {m_base_early['Bias']:+.4f}")

    # 4. Generate or Load Out-Of-Fold Predictions and Residuals
    oof_cache_path = os.path.join(OUT_DIR, "oof_predictions_train.npz")
    oof_cbf_pred, oof_att_pred = generate_or_load_oof_predictions(
        X_tr_n, Y_cbf_tr, Y_att_tr, cbf_mean, cbf_std, att_mean, att_std, oof_cache_path
    )

    res_cbf_tr = Y_cbf_tr - oof_cbf_pred
    res_att_tr = Y_att_tr - oof_att_pred

    # Compute and save OOF residual statistics
    stat_rows = [
        {
            "Parameter": "CBF",
            "OOF_RMSE": float(np.sqrt(np.mean(res_cbf_tr ** 2))),
            "OOF_MAE": float(np.mean(np.abs(res_cbf_tr))),
            "OOF_Mean_Residual": float(np.mean(res_cbf_tr)),
            "OOF_Std_Residual": float(np.std(res_cbf_tr)),
            "OOF_Min_Residual": float(np.min(res_cbf_tr)),
            "OOF_Max_Residual": float(np.max(res_cbf_tr))
        },
        {
            "Parameter": "ATT",
            "OOF_RMSE": float(np.sqrt(np.mean(res_att_tr ** 2))),
            "OOF_MAE": float(np.mean(np.abs(res_att_tr))),
            "OOF_Mean_Residual": float(np.mean(res_att_tr)),
            "OOF_Std_Residual": float(np.std(res_att_tr)),
            "OOF_Min_Residual": float(np.min(res_att_tr)),
            "OOF_Max_Residual": float(np.max(res_att_tr))
        }
    ]
    df_stats = pd.DataFrame(stat_rows)
    stats_csv = os.path.join(OUT_DIR, "oof_residual_statistics.csv")
    df_stats.to_csv(stats_csv, index=False)
    print(f"\nSaved OOF Residual Statistics to {stats_csv}:")
    print(df_stats.to_string(index=False))

    # Standardize residuals for stable ridge solve
    res_cbf_std = float(np.std(res_cbf_tr)) + 1e-6
    res_att_std = float(np.std(res_att_tr)) + 1e-6
    res_cbf_tr_n = (res_cbf_tr / res_cbf_std).astype(np.float32)
    res_att_tr_n = (res_att_tr / res_att_std).astype(np.float32)

    # 5. Extract Feature Sets
    print("\nExtracting Penultimate Latents...")
    with torch.no_grad():
        F_lat_cbf_tr = cbf_net.penultimate_features(torch.from_numpy(X_tr_n).to(device))
        F_lat_cbf_val = cbf_net.penultimate_features(torch.from_numpy(X_val_n).to(device))
        F_lat_att_tr = att_net.penultimate_features(torch.from_numpy(X_tr_n).to(device))
        F_lat_att_val = att_net.penultimate_features(torch.from_numpy(X_val_n).to(device))

    # Permitted auxiliary physics features (Representation D)
    print("Extracting Scaled Physics Features (Rep D)...")
    F_phys_tr_raw = fe.extract_physics_features(X_tr_raw)
    F_phys_val_raw = fe.extract_physics_features(X_val_raw)
    mu_phys, std_phys = fe.fit_feature_scaler(F_phys_tr_raw)
    F_phys_tr = fe.apply_feature_scaler(F_phys_tr_raw, mu_phys, std_phys)
    F_phys_val = fe.apply_feature_scaler(F_phys_val_raw, mu_phys, std_phys)

    F_fusion_cbf_tr = np.hstack([F_lat_cbf_tr, F_phys_tr])
    F_fusion_cbf_val = np.hstack([F_lat_cbf_val, F_phys_val])
    F_fusion_att_tr = np.hstack([F_lat_att_tr, F_phys_tr])
    F_fusion_att_val = np.hstack([F_lat_att_val, F_phys_val])

    # 6. Train and Evaluate the 4 Models Across 5 Seeds
    seeds = [42, 43, 44, 45, 46]
    comparison_records = []

    print("\nTraining and Evaluating 4 Models Across 5 Seeds...")

    # --- Model 1: Existing Feature ELM (Direct Target, Latents Only) ---
    print("Training Model 1: Existing Feature ELM (Direct Target, Latents Only)...")
    cbf_p1_all, att_p1_all = [], []
    for s in seeds:
        ec = GPUChunkedELM(n_hidden=2048, alpha=1.0, activation="elu", seed=s)
        ec.fit_chunked(F_lat_cbf_tr, Y_cbf_tr_std)
        pc = np.clip(ec.predict(F_lat_cbf_val) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
        cbf_p1_all.append(pc)

        ea = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=s)
        ea.fit_chunked(F_lat_att_tr, Y_att_tr_std)
        pa = np.clip(ea.predict(F_lat_att_val) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
        att_p1_all.append(pa)

    p1_cbf_ens = np.mean(cbf_p1_all, axis=0)
    p1_att_ens = np.mean(att_p1_all, axis=0)
    m1_c = calc_metrics(Y_cbf_val, p1_cbf_ens)
    m1_a = calc_metrics(Y_att_val, p1_att_ens)
    m1_early = calc_metrics(Y_att_val[val_early_mask], p1_att_ens[val_early_mask])

    comparison_records.append({
        "Model": "1. Existing Feature ELM",
        "Target_Type": "Direct Parameter",
        "Feature_Input": "Latents Only (A)",
        "Damping_Lambda": 1.0,
        "CBF_RMSE": m1_c["RMSE"], "CBF_MAE": m1_c["MAE"], "CBF_CCC": m1_c["CCC"],
        "ATT_RMSE": m1_a["RMSE"], "ATT_MAE": m1_a["MAE"], "ATT_CCC": m1_a["CCC"],
        "Early_ATT_RMSE": m1_early["RMSE"], "Early_ATT_Bias": m1_early["Bias"], "Early_ATT_CCC": m1_early["CCC"]
    })

    # --- Model 2: Feature-Fusion ELM (Direct Target, Latents + Physics) ---
    print("Training Model 2: Feature-Fusion ELM (Direct Target, Latents + Physics)...")
    cbf_p2_all, att_p2_all = [], []
    for s in seeds:
        ec = GPUChunkedELM(n_hidden=2048, alpha=1.0, activation="elu", seed=s)
        ec.fit_chunked(F_fusion_cbf_tr, Y_cbf_tr_std)
        pc = np.clip(ec.predict(F_fusion_cbf_val) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
        cbf_p2_all.append(pc)

        ea = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=s)
        ea.fit_chunked(F_fusion_att_tr, Y_att_tr_std)
        pa = np.clip(ea.predict(F_fusion_att_val) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
        att_p2_all.append(pa)

    p2_cbf_ens = np.mean(cbf_p2_all, axis=0)
    p2_att_ens = np.mean(att_p2_all, axis=0)
    m2_c = calc_metrics(Y_cbf_val, p2_cbf_ens)
    m2_a = calc_metrics(Y_att_val, p2_att_ens)
    m2_early = calc_metrics(Y_att_val[val_early_mask], p2_att_ens[val_early_mask])

    comparison_records.append({
        "Model": "2. Feature-Fusion ELM",
        "Target_Type": "Direct Parameter",
        "Feature_Input": "Latents + Physics (D)",
        "Damping_Lambda": 1.0,
        "CBF_RMSE": m2_c["RMSE"], "CBF_MAE": m2_c["MAE"], "CBF_CCC": m2_c["CCC"],
        "ATT_RMSE": m2_a["RMSE"], "ATT_MAE": m2_a["MAE"], "ATT_CCC": m2_a["CCC"],
        "Early_ATT_RMSE": m2_early["RMSE"], "Early_ATT_Bias": m2_early["Bias"], "Early_ATT_CCC": m2_early["CCC"]
    })

    # --- Model 3: Out-Of-Fold Residual ELM (Residual Target, Latents Only) ---
    print("Training Model 3: Out-Of-Fold Residual ELM (Latents Only)...")
    cbf_r3_all, att_r3_all = [], []
    for s in seeds:
        ec = GPUChunkedELM(n_hidden=2048, alpha=1.0, activation="tanh", seed=s)
        ec.fit_chunked(F_lat_cbf_tr, res_cbf_tr_n)
        rc = ec.predict(F_lat_cbf_val) * res_cbf_std
        cbf_r3_all.append(rc)

        ea = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=s)
        ea.fit_chunked(F_lat_att_tr, res_att_tr_n)
        ra = ea.predict(F_lat_att_val) * res_att_std
        att_r3_all.append(ra)

    r3_cbf_ens = np.mean(cbf_r3_all, axis=0)
    r3_att_ens = np.mean(att_r3_all, axis=0)

    for lmb in [1.0, 0.5, 0.25, 0.1]:
        p3_cbf = np.clip(p_cbf_dnn_val + lmb * r3_cbf_ens, physics.CBF_MIN, physics.CBF_MAX)
        p3_att = np.clip(p_att_dnn_val + lmb * r3_att_ens, physics.ATT_MIN, physics.ATT_MAX)
        m3_c = calc_metrics(Y_cbf_val, p3_cbf)
        m3_a = calc_metrics(Y_att_val, p3_att)
        m3_early = calc_metrics(Y_att_val[val_early_mask], p3_att[val_early_mask])

        comparison_records.append({
            "Model": f"3. OOF Residual ELM (lambda={lmb})",
            "Target_Type": "OOF Residual",
            "Feature_Input": "Latents Only (A)",
            "Damping_Lambda": lmb,
            "CBF_RMSE": m3_c["RMSE"], "CBF_MAE": m3_c["MAE"], "CBF_CCC": m3_c["CCC"],
            "ATT_RMSE": m3_a["RMSE"], "ATT_MAE": m3_a["MAE"], "ATT_CCC": m3_a["CCC"],
            "Early_ATT_RMSE": m3_early["RMSE"], "Early_ATT_Bias": m3_early["Bias"], "Early_ATT_CCC": m3_early["CCC"]
        })

    # --- Model 4: Feature-Fusion Residual ELM (Residual Target, Latents + Physics) ---
    print("Training Model 4: Feature-Fusion Residual ELM (Latents + Physics)...")
    cbf_r4_all, att_r4_all = [], []
    for s in seeds:
        ec = GPUChunkedELM(n_hidden=2048, alpha=1.0, activation="tanh", seed=s)
        ec.fit_chunked(F_fusion_cbf_tr, res_cbf_tr_n)
        rc = ec.predict(F_fusion_cbf_val) * res_cbf_std
        cbf_r4_all.append(rc)

        ea = GPUChunkedELM(n_hidden=2048, alpha=100.0, activation="tanh", seed=s)
        ea.fit_chunked(F_fusion_att_tr, res_att_tr_n)
        ra = ea.predict(F_fusion_att_val) * res_att_std
        att_r4_all.append(ra)

    r4_cbf_ens = np.mean(cbf_r4_all, axis=0)
    r4_att_ens = np.mean(att_r4_all, axis=0)

    for lmb in [1.0, 0.5, 0.25, 0.1]:
        p4_cbf = np.clip(p_cbf_dnn_val + lmb * r4_cbf_ens, physics.CBF_MIN, physics.CBF_MAX)
        p4_att = np.clip(p_att_dnn_val + lmb * r4_att_ens, physics.ATT_MIN, physics.ATT_MAX)
        m4_c = calc_metrics(Y_cbf_val, p4_cbf)
        m4_a = calc_metrics(Y_att_val, p4_att)
        m4_early = calc_metrics(Y_att_val[val_early_mask], p4_att[val_early_mask])

        comparison_records.append({
            "Model": f"4. Feature-Fusion Residual ELM (lambda={lmb})",
            "Target_Type": "OOF Residual",
            "Feature_Input": "Latents + Physics (D)",
            "Damping_Lambda": lmb,
            "CBF_RMSE": m4_c["RMSE"], "CBF_MAE": m4_c["MAE"], "CBF_CCC": m4_c["CCC"],
            "ATT_RMSE": m4_a["RMSE"], "ATT_MAE": m4_a["MAE"], "ATT_CCC": m4_a["CCC"],
            "Early_ATT_RMSE": m4_early["RMSE"], "Early_ATT_Bias": m4_early["Bias"], "Early_ATT_CCC": m4_early["CCC"]
        })

    # Summary table
    df_comp = pd.DataFrame(comparison_records)
    comp_csv = os.path.join(OUT_DIR, "stage3_residual_comparison.csv")
    df_comp.to_csv(comp_csv, index=False)
    print(f"\nSaved Stage 3 Residual Comparison Table to {comp_csv}:")
    print(df_comp[["Model", "CBF_RMSE", "CBF_MAE", "ATT_RMSE", "ATT_MAE", "Early_ATT_RMSE"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("STAGE 3 RESIDUAL ELM EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()

