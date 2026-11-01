"""
run_reproduction_pipeline.py
Master execution pipeline for paper reproduction:
1. Generates paper-matched training (1M samples) and validation (100k samples) datasets.
2. Normalizes inputs using training statistics only.
3. Reconstructs and trains the 4x64 DNN-G baseline for CBF and ATT with sigmoid-bounded output.
4. Reconstructs and trains the Proposed Residual H-DELM with 15 physics features (79-dim total).
5. Generates the 30,000 independent test samples (10k at SNR 10, 10k at SNR 15, 10k at SNR 20).
6. Evaluates both models on identical test data.
7. Computes MSE, RMSE, MAE, Lin's CCC, and percentage gains.
8. Performs paired bootstrap hypothesis testing.
9. Produces required CSV result tables and publication figures.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

# Paths setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(EXP_DIR)

from experiments.physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    compute_physical_signals, add_rician_noise
)
from models.dnn_g import DNN_G_Net, train_dnn_g, device
from models.residual_hdelm import ResidualHDELM
from simulation.generate_independent_test import generate_independent_test_set
from evaluation.calculate_metrics import compute_all_metrics, compute_percentage_gain
from evaluation.statistical_analysis import paired_bootstrap_comparison
from generate_figures import plot_all_reproduction_figures

def generate_paper_dataset(n_unique, n_snr_levels, seed):
    """
    Paper dataset generation:
    - n_unique CBF-ATT parameter pairs
    - n_snr_levels uniformly spaced in [10.0, 20.0]
    - Total observations: n_unique * n_snr_levels
    """
    print(f"Generating dataset (n_unique={n_unique:,}, n_snr={n_snr_levels}, seed={seed})...")
    rng = np.random.default_rng(seed)
    cbf_unique = rng.uniform(CBF_MIN, CBF_MAX, n_unique).astype(np.float64)
    att_unique = rng.uniform(ATT_MIN, ATT_MAX, n_unique).astype(np.float64)

    clean_unique = compute_physical_signals(PLDS, TAU, cbf_unique, att_unique)
    snr_levels = np.linspace(10.0, 20.0, n_snr_levels, dtype=np.float64)

    cbf_all = np.repeat(cbf_unique, n_snr_levels)
    att_all = np.repeat(att_unique, n_snr_levels)
    clean_all = np.repeat(clean_unique, n_snr_levels, axis=0)
    snr_all = np.tile(snr_levels, n_unique)

    X_raw = add_rician_noise(clean_all, snr_all, rng)
    Y_phys = np.stack([cbf_all, att_all], axis=1).astype(np.float32)
    return X_raw, Y_phys, snr_all

def main():
    print("=" * 80)
    print("REPRODUCING RESEARCH PAPER EXPERIMENTS")
    print("Paper: 'A Residual Hybrid DNN-ELM Model for CBF and ATT Estimation in Multi-Delay ASL MRI'")
    print(f"Compute Device: {device}")
    print("=" * 80)

    results_dir = os.path.join(EXP_DIR, "results")
    models_dir = os.path.join(EXP_DIR, "models")
    figures_dir = os.path.join(EXP_DIR, "figures")
    preds_dir = os.path.join(results_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Dataset Generation: Train (1M) and Val (100k)
    # ---------------------------------------------------------
    t_start = time.time()
    X_tr_raw, Y_tr, snr_tr = generate_paper_dataset(n_unique=10000, n_snr_levels=100, seed=42)
    X_val_raw, Y_val, snr_val = generate_paper_dataset(n_unique=1000, n_snr_levels=100, seed=99)

    # Input normalization: strictly fit on TRAIN only
    print("\nComputing input normalization statistics strictly on Train cohort...")
    X_mean = np.mean(X_tr_raw, axis=0, keepdims=True).astype(np.float32)
    X_std = np.std(X_tr_raw, axis=0, keepdims=True).astype(np.float32)
    X_std = np.where(X_std < 1e-6, 1.0, X_std)

    print(f"  X_mean: {X_mean.squeeze()}")
    print(f"  X_std : {X_std.squeeze()}")

    np.savez_compressed(
        os.path.join(results_dir, "input_normalization_stats.npz"),
        X_mean=X_mean,
        X_std=X_std
    )

    X_tr_n = ((X_tr_raw - X_mean) / X_std).astype(np.float32)
    X_val_n = ((X_val_raw - X_mean) / X_std).astype(np.float32)

    # ---------------------------------------------------------
    # 2. Phase 3: Train Paper-Matched DNN-G Baseline (4x64 for both CBF and ATT)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING AUTHORITATIVE DNN-G BASELINE (4x64 BACKBONES)")
    print("=" * 70)

    # A. CBF Network (4x64, bounds [0, 100])
    cbf_ckpt_path = os.path.join(models_dir, "cbf_dnn_g_best.pth")
    cbf_dnn = DNN_G_Net(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=CBF_MIN, y_max=CBF_MAX)
    print("\nTraining CBFnet (4 hidden layers x 64 neurons)...")
    torch.manual_seed(42)
    cbf_dnn, cbf_hist, cbf_train_sec, cbf_best_val = train_dnn_g(
        cbf_dnn, X_tr_n, Y_tr[:, 0], X_val_n, Y_val[:, 0],
        batch_size=512, max_epochs=200, patience=10, lr=1e-3,
        log_prefix="[CBFnet] "
    )
    torch.save(cbf_dnn.state_dict(), cbf_ckpt_path)
    pd.DataFrame(cbf_hist).to_csv(os.path.join(results_dir, "cbf_dnn_g_training_history.csv"), index=False)
    print(f"Saved CBFnet checkpoint to {cbf_ckpt_path}")

    # B. ATT Network (4x64, bounds [0.5, 3.0])
    att_ckpt_path = os.path.join(models_dir, "att_dnn_g_best.pth")
    att_dnn = DNN_G_Net(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=ATT_MIN, y_max=ATT_MAX)
    print("\nTraining ATTnet (4 hidden layers x 64 neurons)...")
    torch.manual_seed(42)
    att_dnn, att_hist, att_train_sec, att_best_val = train_dnn_g(
        att_dnn, X_tr_n, Y_tr[:, 1], X_val_n, Y_val[:, 1],
        batch_size=512, max_epochs=200, patience=10, lr=1e-3,
        log_prefix="[ATTnet] "
    )
    torch.save(att_dnn.state_dict(), att_ckpt_path)
    pd.DataFrame(att_hist).to_csv(os.path.join(results_dir, "att_dnn_g_training_history.csv"), index=False)
    print(f"Saved ATTnet checkpoint to {att_ckpt_path}")

    # ---------------------------------------------------------
    # 3. Phase 4: Reconstruct Proposed Residual H-DELM
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: FITTING PROPOSED RESIDUAL H-DELM (79 FEATURES, 512 HIDDEN)")
    print("=" * 70)

    print("\nFitting Residual H-DELM for CBF...")
    hdelm_cbf = ResidualHDELM(
        dnn_net=cbf_dnn, y_min=CBF_MIN, y_max=CBF_MAX,
        n_hidden=512, ridge_lambda=1e-2, seed=42
    )
    hdelm_cbf.fit(X_tr_raw, X_tr_n, Y_tr[:, 0])

    print("\nFitting Residual H-DELM for ATT...")
    hdelm_att = ResidualHDELM(
        dnn_net=att_dnn, y_min=ATT_MIN, y_max=ATT_MAX,
        n_hidden=512, ridge_lambda=1e-2, seed=42
    )
    hdelm_att.fit(X_tr_raw, X_tr_n, Y_tr[:, 1])

    # ---------------------------------------------------------
    # 4. Phase 5: Generate Completely Independent 30,000-Observation Test Set
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: GENERATING INDEPENDENT 30,000 TEST OBSERVATIONS")
    print("=" * 70)
    X_test_raw, Y_test, snr_test, test_path = generate_independent_test_set(
        output_dir=preds_dir, seed=2026, samples_per_snr=10000, snr_levels=(10.0, 15.0, 20.0)
    )

    X_test_n = ((X_test_raw - X_mean) / X_std).astype(np.float32)

    # ---------------------------------------------------------
    # 5. Phase 6: Perform Inference on Independent Test Set
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 6: INFERENCE ON INDEPENDENT TEST OBSERVATIONS")
    print("=" * 70)
    print("Running DNN-G and H-DELM inference across 30,000 test observations...")

    cbf_pred_hdelm, cbf_pred_dnn, cbf_res_pred = hdelm_cbf.predict_physical(X_test_raw, X_test_n)
    att_pred_hdelm, att_pred_dnn, att_res_pred = hdelm_att.predict_physical(X_test_raw, X_test_n)

    # Save complete test predictions
    test_preds_path = os.path.join(preds_dir, "reproduction_predictions_30k.npz")
    np.savez_compressed(
        test_preds_path,
        cbf_true=Y_test[:, 0],
        att_true=Y_test[:, 1],
        snr_test=snr_test,
        cbf_dnn=cbf_pred_dnn,
        att_dnn=att_pred_dnn,
        cbf_hdelm=cbf_pred_hdelm,
        att_hdelm=att_pred_hdelm,
        cbf_residual_pred=cbf_res_pred,
        att_residual_pred=att_res_pred
    )
    print(f"Saved complete test predictions to {test_preds_path}")

    # ---------------------------------------------------------
    # 6. Phases 7 & 8: Metrics Calculation & Results Tables
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASES 7 & 8: EVALUATION METRICS & RESULTS TABLES")
    print("=" * 70)

    rows = []
    snr_levels = [10.0, 15.0, 20.0]

    for snr in snr_levels:
        mask = (snr_test == snr)
        
        # Ground truth
        y_cbf_true = Y_test[mask, 0]
        y_att_true = Y_test[mask, 1]

        # Predictions
        p_cbf_dnn = cbf_pred_dnn[mask]
        p_cbf_hdelm = cbf_pred_hdelm[mask]
        p_att_dnn = att_pred_dnn[mask]
        p_att_hdelm = att_pred_hdelm[mask]

        # Calculate metrics
        m_cbf_dnn = compute_all_metrics(y_cbf_true, p_cbf_dnn)
        m_cbf_hdelm = compute_all_metrics(y_cbf_true, p_cbf_hdelm)
        m_att_dnn = compute_all_metrics(y_att_true, p_att_dnn)
        m_att_hdelm = compute_all_metrics(y_att_true, p_att_hdelm)

        rows.append({"SNR": int(snr), "Model": "DNN-G", "Parameter": "CBF", **m_cbf_dnn})
        rows.append({"SNR": int(snr), "Model": "H-DELM", "Parameter": "CBF", **m_cbf_hdelm})
        rows.append({"SNR": int(snr), "Model": "DNN-G", "Parameter": "ATT", **m_att_dnn})
        rows.append({"SNR": int(snr), "Model": "H-DELM", "Parameter": "ATT", **m_att_hdelm})

    metrics_df = pd.DataFrame(rows)

    # Save per-SNR CSVs
    for snr in [10, 15, 20]:
        sub_df = metrics_df[metrics_df["SNR"] == snr][["SNR", "Model", "Parameter", "MSE", "RMSE", "MAE", "Lin_CCC"]]
        csv_p = os.path.join(results_dir, f"results_snr_{snr}.csv")
        sub_df.to_csv(csv_p, index=False)
        print(f"Saved {csv_p}")

    # Complete metrics comparison table
    comp_csv = os.path.join(results_dir, "complete_metrics_comparison.csv")
    metrics_df.to_csv(comp_csv, index=False)
    print(f"Saved {comp_csv}")

    # Print structured comparison table
    print("\n" + "=" * 95)
    print("REQUIRED COMPARISON TABLE (30,000 INDEPENDENT OBSERVATIONS)")
    print("=" * 95)
    print(metrics_df[["SNR", "Model", "Parameter", "MSE", "RMSE", "MAE", "Lin_CCC"]].to_string(index=False))

    # Calculate Percentage Improvement Table
    gain_rows = []
    for snr in [10, 15, 20]:
        for param in ["CBF", "ATT"]:
            d_m = metrics_df[(metrics_df["SNR"] == snr) & (metrics_df["Model"] == "DNN-G") & (metrics_df["Parameter"] == param)].iloc[0]
            h_m = metrics_df[(metrics_df["SNR"] == snr) & (metrics_df["Model"] == "H-DELM") & (metrics_df["Parameter"] == param)].iloc[0]

            rmse_gain = compute_percentage_gain(d_m["RMSE"], h_m["RMSE"])
            mae_gain = compute_percentage_gain(d_m["MAE"], h_m["MAE"])
            gain_rows.append({
                "SNR": snr,
                "Parameter": param,
                "DNN_RMSE": d_m["RMSE"],
                "HDELM_RMSE": h_m["RMSE"],
                "RMSE_Improvement_%": rmse_gain,
                "DNN_MAE": d_m["MAE"],
                "HDELM_MAE": h_m["MAE"],
                "MAE_Improvement_%": mae_gain
            })
    gain_df = pd.DataFrame(gain_rows)
    gain_csv = os.path.join(results_dir, "percentage_improvement_summary.csv")
    gain_df.to_csv(gain_csv, index=False)
    print("\n" + "=" * 95)
    print("PERCENTAGE IMPROVEMENT OF H-DELM RELATIVE TO DNN-G")
    print("=" * 95)
    print(gain_df.round(4).to_string(index=False))

    # ---------------------------------------------------------
    # 7. Phase 10: Statistical Evaluation (Paired Bootstrap)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 10: STATISTICAL EVALUATION (1,000 PAIRED BOOTSTRAPS)")
    print("=" * 70)
    stat_rows = []
    for snr in [10, 15, 20]:
        mask = (snr_test == snr)
        # CBF
        cbf_stat = paired_bootstrap_comparison(
            Y_test[mask, 0], cbf_pred_dnn[mask], cbf_pred_hdelm[mask], n_boot=1000, seed=42
        )
        cbf_stat.update({"SNR": snr, "Parameter": "CBF"})
        stat_rows.append(cbf_stat)

        # ATT
        att_stat = paired_bootstrap_comparison(
            Y_test[mask, 1], att_pred_dnn[mask], att_pred_hdelm[mask], n_boot=1000, seed=42
        )
        att_stat.update({"SNR": snr, "Parameter": "ATT"})
        stat_rows.append(att_stat)

    stat_df = pd.DataFrame(stat_rows)
    stat_csv = os.path.join(results_dir, "statistical_bootstrap_comparison.csv")
    stat_df.to_csv(stat_csv, index=False)
    print(f"Saved statistical bootstrap analysis to {stat_csv}")
    print(stat_df[["SNR", "Parameter", "orig_delta_rmse", "ci_rmse_lower", "ci_rmse_upper", "p_bootstrap_rmse", "orig_delta_mae", "ci_mae_lower", "ci_mae_upper", "p_wilcoxon"]].round(6).to_string(index=False))

    # ---------------------------------------------------------
    # 8. Phase 9: Publication Figures Generation
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 9: GENERATING 9 PUBLICATION-QUALITY FIGURES (PNG 300 DPI + PDF)")
    print("=" * 70)
    preds_dict = {
        "cbf_true": Y_test[:, 0],
        "att_true": Y_test[:, 1],
        "cbf_dnn": cbf_pred_dnn,
        "att_dnn": att_pred_dnn,
        "cbf_hdelm": cbf_pred_hdelm,
        "att_hdelm": att_pred_hdelm
    }
    plot_all_reproduction_figures(metrics_df, preds_dict, figures_dir)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"COMPLETE REPRODUCTION PIPELINE FINISHED IN {total_time:.2f} SECONDS ({total_time/60:.2f} MINUTES)!")
    print(f"Artifacts and tables saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

