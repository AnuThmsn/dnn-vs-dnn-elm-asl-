"""
run_reproduction_pipeline_v2.py
Master execution driver for V2 study:
1. Reuses frozen 4x64 DNN-G backbones and input normalization stats from V1.
2. Generates separate 100k residual-calibration dataset with seed 31415.
3. Conducts hyperparameter sensitivity analysis on ridge lambda grid using calibration-train / val splits.
4. Fits H-DELM V2 exclusively on independent calibration data and standardizes 79-dim features on calibration set.
5. Saves complete H-DELM V2 model artifacts (W, b, beta, feat_mean, feat_std, metadata).
6. Evaluates DNN-G (Group A), H-DELM V1 (Group B), and H-DELM V2 (Group C) on identical 30,000 test observations.
7. Verifies standalone inference engine exact numerical match.
8. Computes 3-way metrics, percentage improvements, paired bootstrap CIs, paired sign-flip permutation tests (B=10,000), Wilcoxon signed-rank tests, and Holm-Bonferroni family-wise error rate corrections.
9. Generates all publication-grade diagnostic figures (PNG 300 DPI + vector PDF).
"""

import os
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
import torch

# Paths setup
V2_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(V2_DIR)
ROOT_DIR = os.path.abspath(os.path.join(V2_DIR, "..", ".."))
sys.path.append(ROOT_DIR)

V1_DIR = os.path.abspath(os.path.join(ROOT_DIR, "experiments", "paper_reproduction_snr10_15_20"))
sys.path.append(V1_DIR)

from experiments.physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    compute_physical_signals, add_rician_noise
)
from models.dnn_g import DNN_G_Net, device
from models.residual_hdelm_v2 import ResidualHDELMV2
from simulation.generate_calibration_dataset import generate_calibration_dataset
from standalone_inference import StandaloneASLEstimator, verify_standalone_predictions
from evaluation.calculate_metrics_v2 import compute_all_metrics, compute_percentage_gain
from evaluation.statistical_testing_v2 import paired_bootstrap_ci, paired_permutation_test, apply_holm_bonferroni
from generate_diagnostics_figures_v2 import plot_all_diagnostics

def log_print(msg, log_file=None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def main():
    t_start = time.time()
    
    # Directories
    results_dir = os.path.join(V2_DIR, "results")
    data_dir = os.path.join(results_dir, "data")
    tables_dir = os.path.join(results_dir, "tables")
    preds_dir = os.path.join(results_dir, "predictions")
    models_dir = os.path.join(V2_DIR, "models")
    figures_dir = os.path.join(V2_DIR, "figures")
    logs_dir = os.path.join(V2_DIR, "logs")

    for d in [results_dir, data_dir, tables_dir, preds_dir, models_dir, figures_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(logs_dir, "execution_v2.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"V2 REPRODUCTION EXPERIMENT EXECUTION LOG\nStarted: {time.ctime()}\n\n")

    log_print("=" * 85, log_path)
    log_print("V2 STUDY: SCIENTIFIC CORRECTION & FINAL VALIDATION OF RESIDUAL H-DELM", log_path)
    log_print("Protocol: Independent Residual Calibration (100k), 3-Way Test Comparison on identical 30k test cohort", log_path)
    log_print(f"Compute Device: {device}", log_path)
    log_print("=" * 85, log_path)

    # -------------------------------------------------------------
    # 1. Verify V1 Assets and Load Frozen DNN Backbones (Group A)
    # -------------------------------------------------------------
    log_print("\n[STEP 1] Verifying V1 assets and loading frozen DNN-G backbones...", log_path)
    cbf_dnn_ckpt = os.path.join(V1_DIR, "models", "cbf_dnn_g_best.pth")
    att_dnn_ckpt = os.path.join(V1_DIR, "models", "att_dnn_g_best.pth")
    norm_stats_path = os.path.join(V1_DIR, "results", "input_normalization_stats.npz")
    test_data_path = os.path.join(V1_DIR, "results", "predictions", "independent_test_30k.npz")
    v1_preds_path = os.path.join(V1_DIR, "results", "predictions", "reproduction_predictions_30k.npz")

    for p in [cbf_dnn_ckpt, att_dnn_ckpt, norm_stats_path, test_data_path, v1_preds_path]:
        assert os.path.exists(p), f"Missing required V1 asset: {p}"
        log_print(f"  Verified asset: {os.path.basename(p)} ({os.path.getsize(p):,} bytes)", log_path)

    norm_data = np.load(norm_stats_path)
    X_mean = norm_data["X_mean"].astype(np.float32)
    X_std = norm_data["X_std"].astype(np.float32)

    cbf_dnn = DNN_G_Net(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=CBF_MIN, y_max=CBF_MAX)
    cbf_dnn.load_state_dict(torch.load(cbf_dnn_ckpt, map_location=device))
    cbf_dnn.to(device)
    cbf_dnn.eval()

    att_dnn = DNN_G_Net(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=ATT_MIN, y_max=ATT_MAX)
    att_dnn.load_state_dict(torch.load(att_dnn_ckpt, map_location=device))
    att_dnn.to(device)
    att_dnn.eval()
    log_print("  Successfully loaded and frozen CBFnet and ATTnet 4x64 backbones.", log_path)

    # -------------------------------------------------------------
    # 2. Independent Residual-Calibration Dataset (100k samples)
    # -------------------------------------------------------------
    log_print("\n[STEP 2] Loading/generating independent residual-calibration dataset (N=100,000, Seed=31415)...", log_path)
    cal_data_path = os.path.join(data_dir, "residual_calibration_100k.npz")
    if not os.path.exists(cal_data_path):
        X_cal_raw, Y_cal_phys, snr_cal, _ = generate_calibration_dataset(
            output_path=cal_data_path, n_samples=100000, seed=31415
        )
    else:
        cal_data = np.load(cal_data_path)
        X_cal_raw = cal_data["X_cal"]
        Y_cal_phys = cal_data["Y_cal"]
        snr_cal = cal_data["snr_cal"]
        log_print(f"  Loaded calibration dataset from disk: {cal_data_path}", log_path)

    X_cal_n = ((X_cal_raw - X_mean) / X_std).astype(np.float32)
    log_print(f"  Calibration data shape: X={X_cal_raw.shape}, Y={Y_cal_phys.shape}", log_path)

    # -------------------------------------------------------------
    # 3. Hyperparameter Sensitivity Analysis on Ridge Lambda Grid
    # -------------------------------------------------------------
    log_print("\n[STEP 3] Running ELM Ridge Lambda Sensitivity Analysis on Calibration Split...", log_path)
    # Split 80,000 calibration-train and 20,000 calibration-validation
    n_cal_tr = 80000
    X_tr_cal_raw, X_val_cal_raw = X_cal_raw[:n_cal_tr], X_cal_raw[n_cal_tr:]
    X_tr_cal_n, X_val_cal_n = X_cal_n[:n_cal_tr], X_cal_n[n_cal_tr:]
    Y_tr_cal_phys, Y_val_cal_phys = Y_cal_phys[:n_cal_tr], Y_cal_phys[n_cal_tr:]

    lambda_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    log_print("  Evaluating lambda grid for CBF ELM...", log_path)
    temp_elm_cbf = ResidualHDELMV2(dnn_net=cbf_dnn, y_min=CBF_MIN, y_max=CBF_MAX, n_hidden=512, seed=42)
    cbf_grid_df, opt_cbf_lambda = temp_elm_cbf.grid_search_lambda(
        X_tr_cal_raw, X_tr_cal_n, Y_tr_cal_phys[:, 0],
        X_val_cal_raw, X_val_cal_n, Y_val_cal_phys[:, 0],
        lambda_grid=lambda_grid
    )
    cbf_grid_df.to_csv(os.path.join(tables_dir, "lambda_sensitivity_cbf.csv"), index=False)

    log_print("  Evaluating lambda grid for ATT ELM...", log_path)
    temp_elm_att = ResidualHDELMV2(dnn_net=att_dnn, y_min=ATT_MIN, y_max=ATT_MAX, n_hidden=512, seed=42)
    att_grid_df, opt_att_lambda = temp_elm_att.grid_search_lambda(
        X_tr_cal_raw, X_tr_cal_n, Y_tr_cal_phys[:, 1],
        X_val_cal_raw, X_val_cal_n, Y_val_cal_phys[:, 1],
        lambda_grid=lambda_grid
    )
    att_grid_df.to_csv(os.path.join(tables_dir, "lambda_sensitivity_att.csv"), index=False)

    log_print(f"  CBF Lambda Grid Analysis:\n{cbf_grid_df.to_string(index=False)}", log_path)
    log_print(f"  ATT Lambda Grid Analysis:\n{att_grid_df.to_string(index=False)}", log_path)
    log_print(f"  Selected CBF lambda={opt_cbf_lambda}, ATT lambda={opt_att_lambda}. Default paper lambda=0.01.", log_path)

    # -------------------------------------------------------------
    # 4. Fit Full H-DELM V2 Models & Save Complete Artifacts
    # -------------------------------------------------------------
    log_print("\n[STEP 4] Fitting H-DELM V2 on full 100k calibration observations & saving artifacts...", log_path)
    hdelm_cbf_v2 = ResidualHDELMV2(
        dnn_net=cbf_dnn, y_min=CBF_MIN, y_max=CBF_MAX,
        n_hidden=512, ridge_lambda=1e-2, seed=42
    )
    hdelm_cbf_v2.fit(X_cal_raw, X_cal_n, Y_cal_phys[:, 0])
    cbf_artifact_path = os.path.join(models_dir, "cbf_hdelm_v2_artifact.npz")
    hdelm_cbf_v2.save_artifact(cbf_artifact_path)

    hdelm_att_v2 = ResidualHDELMV2(
        dnn_net=att_dnn, y_min=ATT_MIN, y_max=ATT_MAX,
        n_hidden=512, ridge_lambda=1e-2, seed=42
    )
    hdelm_att_v2.fit(X_cal_raw, X_cal_n, Y_cal_phys[:, 1])
    att_artifact_path = os.path.join(models_dir, "att_hdelm_v2_artifact.npz")
    hdelm_att_v2.save_artifact(att_artifact_path)

    # -------------------------------------------------------------
    # 5. Load Preserved Independent Test Dataset (30,000 samples)
    # -------------------------------------------------------------
    log_print("\n[STEP 5] Loading preserved independent test cohort (30k observations)...", log_path)
    test_data = np.load(test_data_path)
    X_test_raw = test_data["X_test"].astype(np.float32)
    Y_test_phys = test_data["Y_test"].astype(np.float32)
    snr_test = test_data["snr_test"].astype(np.float64)
    X_test_n = ((X_test_raw - X_mean) / X_std).astype(np.float32)

    # Load V1 predictions
    v1_preds = np.load(v1_preds_path)
    cbf_pred_dnn = v1_preds["cbf_dnn"]
    att_pred_dnn = v1_preds["att_dnn"]
    cbf_pred_v1 = v1_preds["cbf_hdelm"]
    att_pred_v1 = v1_preds["att_hdelm"]

    # Predict with H-DELM V2
    log_print("  Running inference with H-DELM V2 across 30,000 test observations...", log_path)
    cbf_pred_v2, _, cbf_res_v2 = hdelm_cbf_v2.predict_physical(X_test_raw, X_test_n)
    att_pred_v2, _, att_res_v2 = hdelm_att_v2.predict_physical(X_test_raw, X_test_n)

    # Save complete 3-way test predictions
    v2_preds_save_path = os.path.join(preds_dir, "reproduction_predictions_30k_v2.npz")
    np.savez_compressed(
        v2_preds_save_path,
        cbf_true=Y_test_phys[:, 0],
        att_true=Y_test_phys[:, 1],
        snr_test=snr_test,
        cbf_dnn=cbf_pred_dnn,
        att_dnn=att_pred_dnn,
        cbf_hdelm_v1=cbf_pred_v1,
        att_hdelm_v1=att_pred_v1,
        cbf_hdelm_v2=cbf_pred_v2,
        att_hdelm_v2=att_pred_v2,
        cbf_res_v2=cbf_res_v2,
        att_res_v2=att_res_v2
    )
    log_print(f"  Saved 3-way test predictions to {v2_preds_save_path}", log_path)

    # -------------------------------------------------------------
    # 6. Verify Standalone Inference Engine
    # -------------------------------------------------------------
    log_print("\n[STEP 6] Verifying standalone inference engine numerical equivalence...", log_path)
    estimator = StandaloneASLEstimator(
        cbf_dnn_path=cbf_dnn_ckpt,
        att_dnn_path=att_dnn_ckpt,
        cbf_elm_path=cbf_artifact_path,
        att_elm_path=att_artifact_path,
        norm_stats_path=norm_stats_path
    )
    verify_standalone_predictions(test_data_path, v2_preds_save_path, estimator=estimator)
    log_print("  Standalone inference engine validated successfully with exact numerical equality.", log_path)

    # -------------------------------------------------------------
    # 7. Compute 3-Way Metrics Comparison Table (Group A vs B vs C)
    # -------------------------------------------------------------
    log_print("\n[STEP 7] Computing population-consistent evaluation metrics across models and SNR levels...", log_path)
    metric_rows = []
    snr_levels = [10.0, 15.0, 20.0]

    for snr in snr_levels:
        mask = (snr_test == snr)
        y_cbf_t = Y_test_phys[mask, 0]
        y_att_t = Y_test_phys[mask, 1]

        # CBF
        m_cbf_dnn = compute_all_metrics(y_cbf_t, cbf_pred_dnn[mask])
        m_cbf_v1 = compute_all_metrics(y_cbf_t, cbf_pred_v1[mask])
        m_cbf_v2 = compute_all_metrics(y_cbf_t, cbf_pred_v2[mask])

        metric_rows.append({"SNR": int(snr), "Model": "DNN-G", "Parameter": "CBF", **m_cbf_dnn})
        metric_rows.append({"SNR": int(snr), "Model": "H-DELM V1", "Parameter": "CBF", **m_cbf_v1})
        metric_rows.append({"SNR": int(snr), "Model": "H-DELM V2", "Parameter": "CBF", **m_cbf_v2})

        # ATT
        m_att_dnn = compute_all_metrics(y_att_t, att_pred_dnn[mask])
        m_att_v1 = compute_all_metrics(y_att_t, att_pred_v1[mask])
        m_att_v2 = compute_all_metrics(y_att_t, att_pred_v2[mask])

        metric_rows.append({"SNR": int(snr), "Model": "DNN-G", "Parameter": "ATT", **m_att_dnn})
        metric_rows.append({"SNR": int(snr), "Model": "H-DELM V1", "Parameter": "ATT", **m_att_v1})
        metric_rows.append({"SNR": int(snr), "Model": "H-DELM V2", "Parameter": "ATT", **m_att_v2})

    # Overall metrics across all 30k samples
    m_cbf_dnn_all = compute_all_metrics(Y_test_phys[:, 0], cbf_pred_dnn)
    m_cbf_v1_all = compute_all_metrics(Y_test_phys[:, 0], cbf_pred_v1)
    m_cbf_v2_all = compute_all_metrics(Y_test_phys[:, 0], cbf_pred_v2)
    metric_rows.append({"SNR": "All (10-20)", "Model": "DNN-G", "Parameter": "CBF", **m_cbf_dnn_all})
    metric_rows.append({"SNR": "All (10-20)", "Model": "H-DELM V1", "Parameter": "CBF", **m_cbf_v1_all})
    metric_rows.append({"SNR": "All (10-20)", "Model": "H-DELM V2", "Parameter": "CBF", **m_cbf_v2_all})

    m_att_dnn_all = compute_all_metrics(Y_test_phys[:, 1], att_pred_dnn)
    m_att_v1_all = compute_all_metrics(Y_test_phys[:, 1], att_pred_v1)
    m_att_v2_all = compute_all_metrics(Y_test_phys[:, 1], att_pred_v2)
    metric_rows.append({"SNR": "All (10-20)", "Model": "DNN-G", "Parameter": "ATT", **m_att_dnn_all})
    metric_rows.append({"SNR": "All (10-20)", "Model": "H-DELM V1", "Parameter": "ATT", **m_att_v1_all})
    metric_rows.append({"SNR": "All (10-20)", "Model": "H-DELM V2", "Parameter": "ATT", **m_att_v2_all})

    metrics_df = pd.DataFrame(metric_rows)
    comp_csv = os.path.join(tables_dir, "complete_3way_metrics_comparison.csv")
    metrics_df.to_csv(comp_csv, index=False)
    log_print(f"  Saved 3-way metrics comparison to {comp_csv}", log_path)

    # Print nicely formatted comparison table
    log_print("\n" + "=" * 90, log_path)
    log_print("THREE-WAY MODEL COMPARISON TABLE (30,000 INDEPENDENT OBSERVATIONS)", log_path)
    log_print("=" * 90, log_path)
    log_print(metrics_df[["SNR", "Model", "Parameter", "MSE", "RMSE", "MAE", "Bias", "Lin_CCC"]].to_string(index=False), log_path)

    # Percentage gains table
    gain_rows = []
    for snr in [10, 15, 20]:
        for param in ["CBF", "ATT"]:
            d_m = metrics_df[(metrics_df["SNR"] == snr) & (metrics_df["Model"] == "DNN-G") & (metrics_df["Parameter"] == param)].iloc[0]
            v1_m = metrics_df[(metrics_df["SNR"] == snr) & (metrics_df["Model"] == "H-DELM V1") & (metrics_df["Parameter"] == param)].iloc[0]
            v2_m = metrics_df[(metrics_df["SNR"] == snr) & (metrics_df["Model"] == "H-DELM V2") & (metrics_df["Parameter"] == param)].iloc[0]

            v1_rmse_gain = compute_percentage_gain(d_m["RMSE"], v1_m["RMSE"])
            v1_mae_gain = compute_percentage_gain(d_m["MAE"], v1_m["MAE"])

            v2_rmse_gain = compute_percentage_gain(d_m["RMSE"], v2_m["RMSE"])
            v2_mae_gain = compute_percentage_gain(d_m["MAE"], v2_m["MAE"])

            v2_vs_v1_rmse = compute_percentage_gain(v1_m["RMSE"], v2_m["RMSE"])
            v2_vs_v1_mae = compute_percentage_gain(v1_m["MAE"], v2_m["MAE"])

            gain_rows.append({
                "SNR": snr,
                "Parameter": param,
                "DNN_RMSE": d_m["RMSE"],
                "V1_RMSE": v1_m["RMSE"],
                "V2_RMSE": v2_m["RMSE"],
                "V1_Gain_RMSE_%": v1_rmse_gain,
                "V2_Gain_RMSE_%": v2_rmse_gain,
                "V2_vs_V1_RMSE_%": v2_vs_v1_rmse,
                "DNN_MAE": d_m["MAE"],
                "V1_MAE": v1_m["MAE"],
                "V2_MAE": v2_m["MAE"],
                "V1_Gain_MAE_%": v1_mae_gain,
                "V2_Gain_MAE_%": v2_mae_gain,
                "V2_vs_V1_MAE_%": v2_vs_v1_mae
            })
    gain_df = pd.DataFrame(gain_rows)
    gain_csv = os.path.join(tables_dir, "percentage_improvement_summary_v2.csv")
    gain_df.to_csv(gain_csv, index=False)
    log_print(f"  Saved percentage improvements summary to {gain_csv}", log_path)

    log_print("\n" + "=" * 90, log_path)
    log_print("PERCENTAGE IMPROVEMENTS RELATIVE TO DNN-G AND V1 COMPARISON", log_path)
    log_print("=" * 90, log_path)
    log_print(gain_df[["SNR", "Parameter", "V1_Gain_RMSE_%", "V2_Gain_RMSE_%", "V2_vs_V1_RMSE_%", "V1_Gain_MAE_%", "V2_Gain_MAE_%", "V2_vs_V1_MAE_%"]].round(3).to_string(index=False), log_path)

    # -------------------------------------------------------------
    # 8. Rigorous Paired Hypothesis Testing & Holm-Bonferroni Correction
    # -------------------------------------------------------------
    log_print("\n[STEP 8] Conducting paired bootstrap CIs, paired permutation tests (B=10,000), Wilcoxon tests, and Holm-Bonferroni adjustment...", log_path)
    stat_rows_v2_vs_dnn = []
    stat_rows_v2_vs_v1 = []

    for snr in [10, 15, 20]:
        mask = (snr_test == snr)
        
        # --- A. CBF: V2 vs DNN-G ---
        y_cbf_t = Y_test_phys[mask, 0]
        boot_cbf = paired_bootstrap_ci(y_cbf_t, cbf_pred_dnn[mask], cbf_pred_v2[mask], n_boot=1000, seed=42)
        perm_cbf = paired_permutation_test(y_cbf_t, cbf_pred_dnn[mask], cbf_pred_v2[mask], n_permutations=10000, seed=42)
        stat_rows_v2_vs_dnn.append({
            "SNR": snr,
            "Parameter": "CBF",
            "Comparison": "H-DELM V2 vs DNN-G",
            **boot_cbf,
            **perm_cbf
        })

        # --- B. ATT: V2 vs DNN-G ---
        y_att_t = Y_test_phys[mask, 1]
        boot_att = paired_bootstrap_ci(y_att_t, att_pred_dnn[mask], att_pred_v2[mask], n_boot=1000, seed=42)
        perm_att = paired_permutation_test(y_att_t, att_pred_dnn[mask], att_pred_v2[mask], n_permutations=10000, seed=42)
        stat_rows_v2_vs_dnn.append({
            "SNR": snr,
            "Parameter": "ATT",
            "Comparison": "H-DELM V2 vs DNN-G",
            **boot_att,
            **perm_att
        })

        # --- C. V2 vs V1 directly ---
        boot_cbf_v1v2 = paired_bootstrap_ci(y_cbf_t, cbf_pred_v1[mask], cbf_pred_v2[mask], n_boot=1000, seed=42)
        perm_cbf_v1v2 = paired_permutation_test(y_cbf_t, cbf_pred_v1[mask], cbf_pred_v2[mask], n_permutations=10000, seed=42)
        stat_rows_v2_vs_v1.append({
            "SNR": snr,
            "Parameter": "CBF",
            "Comparison": "H-DELM V2 vs H-DELM V1",
            **boot_cbf_v1v2,
            **perm_cbf_v1v2
        })

        boot_att_v1v2 = paired_bootstrap_ci(y_att_t, att_pred_v1[mask], att_pred_v2[mask], n_boot=1000, seed=42)
        perm_att_v1v2 = paired_permutation_test(y_att_t, att_pred_v1[mask], att_pred_v2[mask], n_permutations=10000, seed=42)
        stat_rows_v2_vs_v1.append({
            "SNR": snr,
            "Parameter": "ATT",
            "Comparison": "H-DELM V2 vs H-DELM V1",
            **boot_att_v1v2,
            **perm_att_v1v2
        })

    # Apply Holm-Bonferroni correction across the 6 primary comparisons
    stat_v2_dnn_df = pd.DataFrame(stat_rows_v2_vs_dnn)
    stat_v2_dnn_df = apply_holm_bonferroni(stat_v2_dnn_df, pval_col="p_perm_abs", adjusted_col="p_perm_abs_holm")
    stat_v2_dnn_df = apply_holm_bonferroni(stat_v2_dnn_df, pval_col="p_perm_sq", adjusted_col="p_perm_sq_holm")
    stat_v2_dnn_df = apply_holm_bonferroni(stat_v2_dnn_df, pval_col="wilcoxon_pvalue", adjusted_col="p_wilcoxon_holm")

    stat_v2_v1_df = pd.DataFrame(stat_rows_v2_vs_v1)
    stat_v2_v1_df = apply_holm_bonferroni(stat_v2_v1_df, pval_col="p_perm_abs", adjusted_col="p_perm_abs_holm")
    stat_v2_v1_df = apply_holm_bonferroni(stat_v2_v1_df, pval_col="p_perm_sq", adjusted_col="p_perm_sq_holm")
    stat_v2_v1_df = apply_holm_bonferroni(stat_v2_v1_df, pval_col="wilcoxon_pvalue", adjusted_col="p_wilcoxon_holm")

    all_stats_df = pd.concat([stat_v2_dnn_df, stat_v2_v1_df], ignore_index=True)
    stats_csv = os.path.join(tables_dir, "statistical_hypothesis_testing_v2.csv")
    all_stats_df.to_csv(stats_csv, index=False)
    log_print(f"  Saved statistical testing results to {stats_csv}", log_path)

    log_print("\n" + "=" * 95, log_path)
    log_print("STATISTICAL SIGNIFICANCE SUMMARY (PAIRED PERMUTATION & BOOTSTRAP WITH HOLM-BONFERRONI)", log_path)
    log_print("=" * 95, log_path)
    cols_to_show = ["SNR", "Parameter", "Comparison", "orig_delta_rmse", "ci_rmse_lower", "ci_rmse_upper",
                    "p_perm_sq", "p_perm_sq_holm", "orig_delta_mae", "ci_mae_lower", "ci_mae_upper",
                    "p_perm_abs", "p_perm_abs_holm", "p_perm_abs_holm_sig"]
    log_print(all_stats_df[cols_to_show].round(5).to_string(index=False), log_path)

    # -------------------------------------------------------------
    # 9. Additional Diagnostic Distribution Calculations & Figures
    # -------------------------------------------------------------
    log_print("\n[STEP 9] Extracting diagnostic distributions and plotting 8 publication figures...", log_path)
    # Generate training predictions for diagnostics (1M or subset 100k)
    log_print("  Computing sample DNN training residuals for diagnostics...", log_path)
    rng_diag = np.random.default_rng(42)
    cbf_tr_sample = rng_diag.uniform(CBF_MIN, CBF_MAX, 100000).astype(np.float64)
    att_tr_sample = rng_diag.uniform(ATT_MIN, ATT_MAX, 100000).astype(np.float64)
    snr_tr_sample = rng_diag.uniform(10.0, 20.0, 100000).astype(np.float64)
    clean_tr_sample = compute_physical_signals(PLDS, TAU, cbf_tr_sample, att_tr_sample)
    noisy_x_tr_sample = add_rician_noise(clean_tr_sample, snr_tr_sample, rng_diag)
    x_tr_sample_n = ((noisy_x_tr_sample - X_mean) / X_std).astype(np.float32)

    dnn_cbf_tr_pred = cbf_dnn.predict_physical(x_tr_sample_n)
    dnn_att_tr_pred = att_dnn.predict_physical(x_tr_sample_n)

    r_cbf_train = (cbf_tr_sample - dnn_cbf_tr_pred).astype(np.float32)
    r_att_train = (att_tr_sample - dnn_att_tr_pred).astype(np.float32)

    # Calibration residuals
    dnn_cbf_cal_pred = cbf_dnn.predict_physical(X_cal_n)
    dnn_att_cal_pred = att_dnn.predict_physical(X_cal_n)
    r_cbf_cal = (Y_cal_phys[:, 0] - dnn_cbf_cal_pred).astype(np.float32)
    r_att_cal = (Y_cal_phys[:, 1] - dnn_att_cal_pred).astype(np.float32)

    # H-DELM corrections
    corr_cbf_cal = hdelm_cbf_v2.predict_residual(X_cal_raw, X_cal_n)
    corr_att_cal = hdelm_att_v2.predict_residual(X_cal_raw, X_cal_n)

    diag_data = {
        "r_cbf_train": r_cbf_train,
        "r_att_train": r_att_train,
        "r_cbf_cal": r_cbf_cal,
        "r_att_cal": r_att_cal,
        "corr_cbf_cal": corr_cbf_cal,
        "corr_att_cal": corr_att_cal,
        "corr_cbf_test": cbf_res_v2,
        "corr_att_test": att_res_v2,
        "cbf_true_test": Y_test_phys[:, 0],
        "att_true_test": Y_test_phys[:, 1],
        "cbf_dnn_test": cbf_pred_dnn,
        "att_dnn_test": att_pred_dnn,
        "cbf_v1_test": cbf_pred_v1,
        "att_v1_test": att_pred_v1,
        "cbf_v2_test": cbf_pred_v2,
        "att_v2_test": att_pred_v2
    }

    # Plot all diagnostics
    plot_all_diagnostics(diag_data, metrics_df, cbf_grid_df, att_grid_df, figures_dir)
    log_print(f"  All diagnostic figures successfully rendered in: {figures_dir}", log_path)

    # -------------------------------------------------------------
    # 10. Summary Verification of Numerical Integrity
    # -------------------------------------------------------------
    log_print("\n[STEP 10] Checking numerical integrity and physiological bounds...", log_path)
    for name, arr, bounds in [
        ("CBF DNN-G", cbf_pred_dnn, (0.0, 100.0)),
        ("CBF H-DELM V1", cbf_pred_v1, (0.0, 100.0)),
        ("CBF H-DELM V2", cbf_pred_v2, (0.0, 100.0)),
        ("ATT DNN-G", att_pred_dnn, (0.5, 3.0)),
        ("ATT H-DELM V1", att_pred_v1, (0.5, 3.0)),
        ("ATT H-DELM V2", att_pred_v2, (0.5, 3.0)),
    ]:
        n_nans = int(np.isnan(arr).sum())
        n_infs = int(np.isinf(arr).sum())
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        in_bounds = (min_v >= bounds[0] - 1e-5) and (max_v <= bounds[1] + 1e-5)
        log_print(f"  {name:<15}: NaNs={n_nans}, Infs={n_infs}, Min={min_v:.3f}, Max={max_v:.3f} | Bounds {bounds} respected: {in_bounds}", log_path)
        assert n_nans == 0 and n_infs == 0, f"Detected invalid values in {name}!"
        assert in_bounds, f"Out-of-bounds predictions detected in {name}!"

    total_time = time.time() - t_start
    log_print("\n" + "=" * 85, log_path)
    log_print(f"V2 REPRODUCTION EXPERIMENT COMPLETED IN {total_time:.2f}s ({total_time/60:.2f} min)", log_path)
    log_print(f"Results Directory : {results_dir}", log_path)
    log_print(f"Figures Directory : {figures_dir}", log_path)
    log_print(f"Models Directory  : {models_dir}", log_path)
    log_print("=" * 85, log_path)

if __name__ == "__main__":
    main()

