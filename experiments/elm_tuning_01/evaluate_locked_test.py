"""
evaluate_locked_test.py
Stage 5: One Final Locked-Test Evaluation of Tuned ELM vs. Baseline A DNN and Existing ELM.
- STRICT RULE: Hyperparameter configurations are frozen from Stage 2 & 3 validation sweeps.
- Evaluates on the held-out test cohort (N=100,000).
- Compares:
    1. Baseline A DNN (Experiment G DNN-G)
    2. Historical Ensemble Feature ELM (5 seeds, H=512, alpha=1e-2, tanh)
    3. Tuned ELM Single Best Seed
    4. Tuned ELM 5-Seed Ensemble
    5. Tuned ELM 10-Seed Ensemble
- Computes:
    - Overall RMSE, MAE, Bias, R2, CCC
    - Early ATT ([0.5, 1.0) s) diagnostic metrics
    - Stratified metrics across SNR levels: [20, 30, 40, 60, inf]
    - Percentage improvement relative to Baseline A DNN and Historical Feature ELM
- Saves:
    - final_test_metrics.csv
    - snr_stratified_test_metrics.csv
    - final_test_predictions.npz (compact predictions for auditing)
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

sys.path.append(EXP_DIR)
sys.path.append(os.path.join(EXP_DIR, "elm_models"))
import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

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

def run_evaluation():
    print("=" * 80)
    print("STAGE 5: LOCKED TEST SET EVALUATION (N=100,000)")
    print("=" * 80)

    # 1. Load Normalization Stats & Checkpoints
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
    x_te_mmap = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    y_te_mmap = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    snr_te = np.load(os.path.join(CACHE_DIR, "SNR_test.npy"))

    X_te_n = ((x_te_mmap[:] - X_mean) / X_std).astype(np.float32)
    X_tr_n = ((x_tr_mmap[:] - X_mean) / X_std).astype(np.float32)

    Y_cbf_te = y_te_mmap[:, 0].astype(np.float64)
    Y_att_te = y_te_mmap[:, 1].astype(np.float64)

    Y_cbf_tr_std = ((y_tr_mmap[:, 0] - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_tr_std = ((y_tr_mmap[:, 1] - att_mean) / att_std).astype(np.float32)

    early_mask_te = (Y_att_te >= 0.5) & (Y_att_te < 1.0)

    # 3. Extract Penultimate Features to Disk Cache
    f_cbf_tr_path = os.path.join(FEAT_CACHE_DIR, "F_cbf_train.npy")
    f_cbf_te_path = os.path.join(FEAT_CACHE_DIR, "F_cbf_test.npy")
    F_cbf_tr = extract_features_to_mmap(cbf_net, X_tr_n, f_cbf_tr_path)
    F_cbf_te = extract_features_to_mmap(cbf_net, X_te_n, f_cbf_te_path)

    f_att_tr_path = os.path.join(FEAT_CACHE_DIR, "F_att_train.npy")
    f_att_te_path = os.path.join(FEAT_CACHE_DIR, "F_att_test.npy")
    F_att_tr = extract_features_to_mmap(att_net, X_tr_n, f_att_tr_path)
    F_att_te = extract_features_to_mmap(att_net, X_te_n, f_att_te_path)

    # 4. Load Frozen Selected Hyperparameters from Validation Sweep
    val_comp_path = os.path.join(OUT_DIR, "validation_comparison.csv")
    if not os.path.exists(val_comp_path):
        raise FileNotFoundError(f"Missing validation comparison: {val_comp_path}")
    df_val = pd.read_csv(val_comp_path)

    sweep_csv = os.path.join(OUT_DIR, "validation_sweep.csv")
    df_sweep = pd.read_csv(sweep_csv)

    # Best validation configuration for CBF
    cbf_best_row = df_sweep[df_sweep["Target"] == "CBF"].sort_values("Val_RMSE").iloc[0]
    best_cbf_nh = int(cbf_best_row["Hidden_Neurons"])
    best_cbf_al = float(cbf_best_row["Alpha"])
    best_cbf_act = str(cbf_best_row["Activation"])
    print(f"\n[Selected CBF Config from Validation]: H={best_cbf_nh}, alpha={best_cbf_al}, act={best_cbf_act}")

    # Best validation configuration for ATT
    att_best_row = df_sweep[df_sweep["Target"] == "ATT"].sort_values("Val_RMSE").iloc[0]
    best_att_nh = int(att_best_row["Hidden_Neurons"])
    best_att_al = float(att_best_row["Alpha"])
    best_att_act = str(att_best_row["Activation"])
    print(f"[Selected ATT Config from Validation]: H={best_att_nh}, alpha={best_att_al}, act={best_att_act}")

    # 5. Load Historical Predictions for Baseline A DNN and Historical Feature ELM
    all_preds_path = os.path.join(EXP_DIR, "elm_models", "all_test_predictions.npz")
    all_hist_preds = np.load(all_preds_path)
    pred_cbf_dnn = all_hist_preds["baseline_a_dnn_g_cbf_test"].astype(np.float64)
    pred_att_dnn = all_hist_preds["baseline_a_dnn_g_att_test"].astype(np.float64)
    pred_cbf_hist_elm5 = all_hist_preds["ensemble_feature_elm_5_seeds_cbf_test"].astype(np.float64)
    pred_att_hist_elm5 = all_hist_preds["ensemble_feature_elm_5_seeds_att_test"].astype(np.float64)

    pred_save_path = os.path.join(OUT_DIR, "final_test_predictions.npz")
    if os.path.exists(pred_save_path):
        print(f"\nLoading existing saved test predictions from: {pred_save_path}")
        saved_preds = np.load(pred_save_path)
        pred_cbf_tuned_single = saved_preds["tuned_elm_cbf_single"].astype(np.float64)
        pred_att_tuned_single = saved_preds["tuned_elm_att_single"].astype(np.float64)
        pred_cbf_tuned_ens5 = saved_preds["tuned_elm_cbf_ens5"].astype(np.float64)
        pred_att_tuned_ens5 = saved_preds["tuned_elm_att_ens5"].astype(np.float64)
        pred_cbf_tuned_ens10 = saved_preds["tuned_elm_cbf_ens10"].astype(np.float64)
        pred_att_tuned_ens10 = saved_preds["tuned_elm_att_ens10"].astype(np.float64)
    else:
        # 6. Fit and Predict Tuned Models on Train Features and Predict on Test
        seeds_10 = list(range(42, 52))

        # Fit Tuned CBF Models across 10 seeds
        print("\nFitting Tuned CBF ELM Models (10 seeds)...")
        tuned_cbf_preds = []
        for s in seeds_10:
            elm = GPUChunkedELM(n_hidden=best_cbf_nh, alpha=best_cbf_al, activation=best_cbf_act, seed=s)
            elm.fit_chunked(F_cbf_tr, Y_cbf_tr_std)
            p = np.clip(elm.predict(F_cbf_te) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX).astype(np.float64)
            tuned_cbf_preds.append(p)
            del elm

        pred_cbf_tuned_single = tuned_cbf_preds[0]
        pred_cbf_tuned_ens5 = np.mean(tuned_cbf_preds[:5], axis=0)
        pred_cbf_tuned_ens10 = np.mean(tuned_cbf_preds, axis=0)

        # Fit Tuned ATT Models across 10 seeds
        print("Fitting Tuned ATT ELM Models (10 seeds)...")
        tuned_att_preds = []
        for s in seeds_10:
            elm = GPUChunkedELM(n_hidden=best_att_nh, alpha=best_att_al, activation=best_att_act, seed=s)
            elm.fit_chunked(F_att_tr, Y_att_tr_std)
            p = np.clip(elm.predict(F_att_te) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX).astype(np.float64)
            tuned_att_preds.append(p)
            del elm

        pred_att_tuned_single = tuned_att_preds[0]
        pred_att_tuned_ens5 = np.mean(tuned_att_preds[:5], axis=0)
        pred_att_tuned_ens10 = np.mean(tuned_att_preds, axis=0)

        np.savez_compressed(
            pred_save_path,
            tuned_elm_cbf_single=pred_cbf_tuned_single.astype(np.float32),
            tuned_elm_att_single=pred_att_tuned_single.astype(np.float32),
            tuned_elm_cbf_ens5=pred_cbf_tuned_ens5.astype(np.float32),
            tuned_elm_att_ens5=pred_att_tuned_ens5.astype(np.float32),
            tuned_elm_cbf_ens10=pred_cbf_tuned_ens10.astype(np.float32),
            tuned_elm_att_ens10=pred_att_tuned_ens10.astype(np.float32)
        )
        print(f"Saved predictions array to: {pred_save_path}")

    # 7. Compute Test Metrics
    models = [
        ("Baseline A (DNN-G)", pred_cbf_dnn, pred_att_dnn, "StandardizedNet (Frozen)"),
        ("Historical Feature ELM (5 seeds)", pred_cbf_hist_elm5, pred_att_hist_elm5, "H=512, alpha=1e-2, tanh (5 seeds)"),
        ("Tuned ELM Single Best", pred_cbf_tuned_single, pred_att_tuned_single, f"CBF:[H={best_cbf_nh},{best_cbf_al:.1e},{best_cbf_act}] ATT:[H={best_att_nh},{best_att_al:.1e},{best_att_act}] (seed 42)"),
        ("Tuned ELM Ensemble (5 seeds)", pred_cbf_tuned_ens5, pred_att_tuned_ens5, f"CBF:[H={best_cbf_nh},{best_cbf_al:.1e},{best_cbf_act}] ATT:[H={best_att_nh},{best_att_al:.1e},{best_att_act}] (5 seeds)"),
        ("Tuned ELM Ensemble (10 seeds)", pred_cbf_tuned_ens10, pred_att_tuned_ens10, f"CBF:[H={best_cbf_nh},{best_cbf_al:.1e},{best_cbf_act}] ATT:[H={best_att_nh},{best_att_al:.1e},{best_att_act}] (10 seeds)")
    ]

    base_cbf_rmse = calc_metrics(Y_cbf_te, pred_cbf_dnn)["RMSE"]
    base_att_rmse = calc_metrics(Y_att_te, pred_att_dnn)["RMSE"]
    hist_cbf_rmse = calc_metrics(Y_cbf_te, pred_cbf_hist_elm5)["RMSE"]
    hist_att_rmse = calc_metrics(Y_att_te, pred_att_hist_elm5)["RMSE"]

    records = []
    for name, p_cbf, p_att, cfg_desc in models:
        m_cbf = calc_metrics(Y_cbf_te, p_cbf)
        m_att = calc_metrics(Y_att_te, p_att)
        m_early_att = calc_metrics(Y_att_te[early_mask_te], p_att[early_mask_te])

        records.append({
            "Model": name,
            "Configuration": cfg_desc,
            "CBF_RMSE": m_cbf["RMSE"],
            "CBF_MAE": m_cbf["MAE"],
            "CBF_CCC": m_cbf["CCC"],
            "CBF_Gain_vs_DNN_%": 100.0 * (base_cbf_rmse - m_cbf["RMSE"]) / base_cbf_rmse,
            "CBF_Gain_vs_HistELM_%": 100.0 * (hist_cbf_rmse - m_cbf["RMSE"]) / hist_cbf_rmse,
            "ATT_RMSE": m_att["RMSE"],
            "ATT_MAE": m_att["MAE"],
            "ATT_CCC": m_att["CCC"],
            "ATT_Gain_vs_DNN_%": 100.0 * (base_att_rmse - m_att["RMSE"]) / base_att_rmse,
            "ATT_Gain_vs_HistELM_%": 100.0 * (hist_att_rmse - m_att["RMSE"]) / hist_att_rmse,
            "Early_ATT_RMSE": m_early_att["RMSE"],
            "Early_ATT_Bias": m_early_att["Bias"],
            "Early_ATT_CCC": m_early_att["CCC"]
        })

    df_results = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "final_test_metrics.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved final locked-test metrics to: {out_csv}")
    print(df_results[["Model", "CBF_RMSE", "CBF_Gain_vs_DNN_%", "ATT_RMSE", "ATT_Gain_vs_DNN_%", "Early_ATT_RMSE"]].to_string(index=False))

    # 8. SNR Stratified Metrics across all 12 actual test SNR levels
    snr_levels = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, np.inf]
    snr_records = []
    for snr_val in snr_levels:
        mask = np.isinf(snr_te) if np.isinf(snr_val) else (np.abs(snr_te - snr_val) < 1e-3)
        snr_lbl = "inf" if np.isinf(snr_val) else f"{int(snr_val)}"

        for name, p_cbf, p_att, _ in models:
            m_cbf = calc_metrics(Y_cbf_te[mask], p_cbf[mask])
            m_att = calc_metrics(Y_att_te[mask], p_att[mask])
            snr_records.append({
                "SNR": snr_lbl, "Model": name,
                "CBF_RMSE": m_cbf["RMSE"], "CBF_MAE": m_cbf["MAE"], "CBF_CCC": m_cbf["CCC"],
                "ATT_RMSE": m_att["RMSE"], "ATT_MAE": m_att["MAE"], "ATT_CCC": m_att["CCC"]
            })

    df_snr = pd.DataFrame(snr_records)
    snr_csv = os.path.join(OUT_DIR, "snr_stratified_test_metrics.csv")
    df_snr.to_csv(snr_csv, index=False)
    print(f"Saved SNR stratified test metrics to: {snr_csv}")

    # 9. Save test predictions to compact npz
    pred_save_path = os.path.join(OUT_DIR, "final_test_predictions.npz")
    np.savez_compressed(
        pred_save_path,
        tuned_elm_cbf_single=pred_cbf_tuned_single.astype(np.float32),
        tuned_elm_att_single=pred_att_tuned_single.astype(np.float32),
        tuned_elm_cbf_ens5=pred_cbf_tuned_ens5.astype(np.float32),
        tuned_elm_att_ens5=pred_att_tuned_ens5.astype(np.float32),
        tuned_elm_cbf_ens10=pred_cbf_tuned_ens10.astype(np.float32),
        tuned_elm_att_ens10=pred_att_tuned_ens10.astype(np.float32)
    )
    print(f"Saved predictions array to: {pred_save_path}")
    print("=" * 80)
    print("STAGE 5 LOCKED TEST EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()

