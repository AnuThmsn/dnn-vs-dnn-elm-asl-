"""
run_stage4_calibration.py
Stage 4: Post-Hoc Calibration, Affine Correction, Shrinkage, and Non-Linear Calibration
of DNN-ELM predictions under fixed 4-PLD protocol.
Zero test leakage: calibration parameters fitted on 50k validation split and evaluated
on held-out 50k validation split.
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
OUT_DIR = os.path.join(EXP_DIR, "reference_calibration_03")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.append(EXP_DIR)
sys.path.append(os.path.join(EXP_DIR, "elm_models"))

import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

def main():
    print("=" * 80)
    print("EXPERIMENT 3 - STAGE 4: CALIBRATION & SHRINKAGE OF DNN-ELM PREDICTIONS")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Normalization & Models
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    att_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device, weights_only=False))
    att_net.eval()

    cbf_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device, weights_only=False))
    cbf_net.eval()

    # 2. Datasets
    # 100k training sample for fitting Feature ELM
    N_tr = 100_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    X_train_raw = np.array(x_tr_mmap[idx_tr], dtype=np.float32)
    Y_train_raw = np.array(y_tr_mmap[idx_tr], dtype=np.float32)
    X_train_dnn = (X_train_raw - X_mean) / X_std

    N_val = len(x_val_mmap)
    X_val_raw = np.array(x_val_mmap[:], dtype=np.float32)
    Y_val_raw = np.array(y_val_mmap[:], dtype=np.float32)
    X_val_dnn = (X_val_raw - X_mean) / X_std

    # 3. Fit Historical (H=512, alpha=1e-2) and Tuned (H=2048, alpha=10/1000) Feature ELMs
    print("\nExtracting Training Latents & Fitting Feature ELMs...")
    chunk_sz = 20_000
    F_tr_cbf = []
    F_tr_att = []
    for i in range(0, N_tr, chunk_sz):
        F_tr_cbf.append(cbf_net.penultimate_features(X_train_dnn[i:i+chunk_sz]))
        F_tr_att.append(att_net.penultimate_features(X_train_dnn[i:i+chunk_sz]))
    F_tr_cbf = np.vstack(F_tr_cbf)
    F_tr_att = np.vstack(F_tr_att)

    y_tr_cbf_std = (Y_train_raw[:, 0] - cbf_mean) / cbf_std
    y_tr_att_std = (Y_train_raw[:, 1] - att_mean) / att_std

    # 5 seeds for Tuned Feature ELM
    cbf_elms = []
    att_elms = []
    for seed in [42, 101, 2024, 7, 99]:
        elm_c = GPUChunkedELM(n_hidden=2048, alpha=10.0, activation="tanh", seed=seed)
        elm_c.fit_chunked(F_tr_cbf, y_tr_cbf_std)
        cbf_elms.append(elm_c)

        elm_a = GPUChunkedELM(n_hidden=2048, alpha=1000.0, activation="tanh", seed=seed)
        elm_a.fit_chunked(F_tr_att, y_tr_att_std)
        att_elms.append(elm_a)

    del F_tr_cbf, F_tr_att
    torch.cuda.empty_cache()

    # 4. Generate Predictions on full Validation Set
    print("\nGenerating Base Predictions on Validation (N=100,000)...")
    cbf_pred_dnn = cbf_net.predict_physical(X_val_dnn)
    att_pred_dnn = att_net.predict_physical(X_val_dnn)

    F_val_cbf = []
    F_val_att = []
    for i in range(0, N_val, chunk_sz):
        F_val_cbf.append(cbf_net.penultimate_features(X_val_dnn[i:i+chunk_sz]))
        F_val_att.append(att_net.penultimate_features(X_val_dnn[i:i+chunk_sz]))
    F_val_cbf = np.vstack(F_val_cbf)
    F_val_att = np.vstack(F_val_att)

    cbf_elm_preds = []
    for elm in cbf_elms:
        cbf_elm_preds.append(np.clip(elm.predict(F_val_cbf) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX))
    cbf_pred_elm = np.mean(cbf_elm_preds, axis=0)

    att_elm_preds = []
    for elm in att_elms:
        att_elm_preds.append(np.clip(elm.predict(F_val_att) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX))
    att_pred_elm = np.mean(att_elm_preds, axis=0)

    del F_val_cbf, F_val_att, cbf_net, att_net
    torch.cuda.empty_cache()

    # 5. Split Validation into Calibration (50k) and Held-Out Validation Evaluation (50k)
    # Split index: first 50k = calibrate, second 50k = evaluate
    N_half = 50_000
    idx_cal = np.arange(0, N_half)
    idx_eval = np.arange(N_half, N_val)

    y_cal_cbf = Y_val_raw[idx_cal, 0]
    y_cal_att = Y_val_raw[idx_cal, 1]
    y_eval_cbf = Y_val_raw[idx_eval, 0]
    y_eval_att = Y_val_raw[idx_eval, 1]

    dnn_cal_cbf = cbf_pred_dnn[idx_cal]
    dnn_cal_att = att_pred_dnn[idx_cal]
    elm_cal_cbf = cbf_pred_elm[idx_cal]
    elm_cal_att = att_pred_elm[idx_cal]

    dnn_eval_cbf = cbf_pred_dnn[idx_eval]
    dnn_eval_att = att_pred_dnn[idx_eval]
    elm_eval_cbf = cbf_pred_elm[idx_eval]
    elm_eval_att = att_pred_elm[idx_eval]

    # Baseline metrics on held-out eval split
    m_base_dnn_cbf = calc_metrics(y_eval_cbf, dnn_eval_cbf)
    m_base_dnn_att = calc_metrics(y_eval_att, dnn_eval_att)
    m_base_elm_cbf = calc_metrics(y_eval_cbf, elm_eval_cbf)
    m_base_elm_att = calc_metrics(y_eval_att, elm_eval_att)

    print("\nBaseline Metrics on Held-Out Validation Split (N=50,000):")
    print(f"  DNN CBF RMSE: {m_base_dnn_cbf['RMSE']:.4f}, Bias: {m_base_dnn_cbf['Bias']:+.3f}")
    print(f"  DNN ATT RMSE: {m_base_dnn_att['RMSE']:.4f}, Bias: {m_base_dnn_att['Bias']:+.3f}")
    print(f"  ELM CBF RMSE: {m_base_elm_cbf['RMSE']:.4f}, Bias: {m_base_elm_cbf['Bias']:+.3f}")
    print(f"  ELM ATT RMSE: {m_base_elm_att['RMSE']:.4f}, Bias: {m_base_elm_att['Bias']:+.3f}")

    # =========================================================================
    # FIT CALIBRATION MODELS STRICTLY ON CALIBRATION SPLIT (N=50,000)
    # =========================================================================
    print("\nFitting Calibration Models on Calibration Split...")
    calib_params = {}

    # Method 1: Affine Calibration (a * pred + b)
    # CBF DNN
    p_cbf_dnn = np.polyfit(dnn_cal_cbf, y_cal_cbf, deg=1)
    # CBF ELM
    p_cbf_elm = np.polyfit(elm_cal_cbf, y_cal_cbf, deg=1)
    # ATT DNN
    p_att_dnn = np.polyfit(dnn_cal_att, y_cal_att, deg=1)
    # ATT ELM
    p_att_elm = np.polyfit(elm_cal_att, y_cal_att, deg=1)

    calib_params["affine_cbf_dnn"] = {"slope": float(p_cbf_dnn[0]), "intercept": float(p_cbf_dnn[1])}
    calib_params["affine_cbf_elm"] = {"slope": float(p_cbf_elm[0]), "intercept": float(p_cbf_elm[1])}
    calib_params["affine_att_dnn"] = {"slope": float(p_att_dnn[0]), "intercept": float(p_att_dnn[1])}
    calib_params["affine_att_elm"] = {"slope": float(p_att_elm[0]), "intercept": float(p_att_elm[1])}

    # Method 2: Validation-Selected Shrinkage gamma in [0, 1]
    # blend = (1 - gamma) * DNN + gamma * ELM
    gammas = np.linspace(0.0, 1.0, 101)
    cbf_rmses = [np.sqrt(np.mean((((1.0 - g) * dnn_cal_cbf + g * elm_cal_cbf) - y_cal_cbf) ** 2)) for g in gammas]
    att_rmses = [np.sqrt(np.mean((((1.0 - g) * dnn_cal_att + g * elm_cal_att) - y_cal_att) ** 2)) for g in gammas]

    best_gamma_cbf = float(gammas[np.argmin(cbf_rmses)])
    best_gamma_att = float(gammas[np.argmin(att_rmses)])
    calib_params["optimal_shrinkage_gamma"] = {"CBF": best_gamma_cbf, "ATT": best_gamma_att}

    # Method 3: Degree-2 Polynomial Calibration (c2 * x^2 + c1 * x + c0)
    poly_cbf_elm = np.polyfit(elm_cal_cbf, y_cal_cbf, deg=2)
    poly_att_elm = np.polyfit(elm_cal_att, y_cal_att, deg=2)
    calib_params["poly2_cbf_elm"] = [float(c) for c in poly_cbf_elm]
    calib_params["poly2_att_elm"] = [float(c) for c in poly_att_elm]

    # Method 4: Early-ATT Damped Calibration (Smooth sigmoidal transition)
    # In [0.5, 1.0), ATT suffers from +0.36 s positive bias.
    # Model bias residual r(att) = bias_early * (1 - sigmoid((att - 1.0)/0.2))
    early_cal_mask = (y_cal_att < 1.0)
    early_bias_att = float(np.mean(elm_cal_att[early_cal_mask] - y_cal_att[early_cal_mask]))
    calib_params["early_att_residual_correction"] = {
        "early_bias": early_bias_att,
        "transition_point": 1.0,
        "transition_width": 0.2
    }

    with open(os.path.join(OUT_DIR, "stage4_calibration_coefficients.json"), "w") as f:
        json.dump(calib_params, f, indent=2)
    print("Saved stage4_calibration_coefficients.json")

    # =========================================================================
    # EVALUATE ALL METHODS ON HELD-OUT VALIDATION SPLIT (N=50,000)
    # =========================================================================
    print("\nEvaluating Calibrated Models on Held-Out Validation Split...")
    models_to_eval = [
        ("1. Baseline A DNN (Uncalibrated)", dnn_eval_cbf, dnn_eval_att),
        ("2. Tuned Feature ELM (Uncalibrated)", elm_eval_cbf, elm_eval_att),
        ("3. Affine-Calibrated DNN",
         np.clip(p_cbf_dnn[0] * dnn_eval_cbf + p_cbf_dnn[1], physics.CBF_MIN, physics.CBF_MAX),
         np.clip(p_att_dnn[0] * dnn_eval_att + p_att_dnn[1], physics.ATT_MIN, physics.ATT_MAX)),
        ("4. Affine-Calibrated Feature ELM",
         np.clip(p_cbf_elm[0] * elm_eval_cbf + p_cbf_elm[1], physics.CBF_MIN, physics.CBF_MAX),
         np.clip(p_att_elm[0] * elm_eval_att + p_att_elm[1], physics.ATT_MIN, physics.ATT_MAX)),
        ("5. Optimal Shrinkage Blend (gamma*)",
         np.clip((1.0 - best_gamma_cbf) * dnn_eval_cbf + best_gamma_cbf * elm_eval_cbf, physics.CBF_MIN, physics.CBF_MAX),
         np.clip((1.0 - best_gamma_att) * dnn_eval_att + best_gamma_att * elm_eval_att, physics.ATT_MIN, physics.ATT_MAX)),
        ("6. Polynomial-2 Calibrated ELM",
         np.clip(np.polyval(poly_cbf_elm, elm_eval_cbf), physics.CBF_MIN, physics.CBF_MAX),
         np.clip(np.polyval(poly_att_elm, elm_eval_att), physics.ATT_MIN, physics.ATT_MAX)),
        ("7. Early-ATT Sigmoid-Damped ELM",
         elm_eval_cbf,
         np.clip(elm_eval_att - early_bias_att / (1.0 + np.exp((elm_eval_att - 1.0) / 0.15)), physics.ATT_MIN, physics.ATT_MAX))
    ]

    eval_results = []
    early_eval_mask = (y_eval_att >= 0.5) & (y_eval_att < 1.0)

    for m_label, p_cbf, p_att in models_to_eval:
        m_cbf = calc_metrics(y_eval_cbf, p_cbf)
        m_att = calc_metrics(y_eval_att, p_att)

        early_m_att = calc_metrics(y_eval_att[early_eval_mask], p_att[early_eval_mask])
        gain_cbf = (m_base_dnn_cbf["RMSE"] - m_cbf["RMSE"]) / m_base_dnn_cbf["RMSE"] * 100.0
        gain_att = (m_base_dnn_att["RMSE"] - m_att["RMSE"]) / m_base_dnn_att["RMSE"] * 100.0

        print(f"\n{m_label}:")
        print(f"  CBF: RMSE={m_cbf['RMSE']:.4f}, MAE={m_cbf['MAE']:.4f}, Bias={m_cbf['Bias']:+.3f}, Gain={gain_cbf:+.2f}%")
        print(f"  ATT: RMSE={m_att['RMSE']:.4f}, MAE={m_att['MAE']:.4f}, Bias={m_att['Bias']:+.3f}, Gain={gain_att:+.2f}%")
        print(f"  Early-ATT: RMSE={early_m_att['RMSE']:.4f}, Bias={early_m_att['Bias']:+.3f}, CCC={early_m_att['CCC']:.4f}")

        eval_results.append({
            "Model": m_label,
            "CBF_RMSE": m_cbf["RMSE"],
            "CBF_MAE": m_cbf["MAE"],
            "CBF_Bias": m_cbf["Bias"],
            "CBF_CCC": m_cbf["CCC"],
            "CBF_R2": m_cbf["R2"],
            "CBF_Gain_vs_DNN_%": gain_cbf,
            "ATT_RMSE": m_att["RMSE"],
            "ATT_MAE": m_att["MAE"],
            "ATT_Bias": m_att["Bias"],
            "ATT_CCC": m_att["CCC"],
            "ATT_R2": m_att["R2"],
            "ATT_Gain_vs_DNN_%": gain_att,
            "Early_ATT_RMSE": early_m_att["RMSE"],
            "Early_ATT_Bias": early_m_att["Bias"],
            "Early_ATT_CCC": early_m_att["CCC"]
        })

    df_eval = pd.DataFrame(eval_results)
    csv_eval_path = os.path.join(OUT_DIR, "stage4_calibration_results.csv")
    df_eval.to_csv(csv_eval_path, index=False)
    print(f"\nSaved calibration results to: {csv_eval_path}")

    # =========================================================================
    # GENERATE CALIBRATION DIAGNOSTIC PLOTS
    # =========================================================================
    print("\nGenerating Calibration Diagnostic Plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # 1. Shrinkage Profile
    ax1 = axes[0, 0]
    ax1.plot(gammas, cbf_rmses, color="#1f77b4", linewidth=2, label="CBF Validation RMSE")
    ax1.axvline(best_gamma_cbf, color="#1f77b4", linestyle="--", label=f"Best CBF gamma*={best_gamma_cbf:.2f}")
    ax1.set_xlabel("Shrinkage Weight gamma (0=DNN, 1=ELM)", fontweight="bold")
    ax1.set_ylabel("CBF RMSE (mL/100g/min)", color="#1f77b4", fontweight="bold")
    ax1.set_title("Validation Shrinkage Profile: CBF & ATT", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_t = ax1.twinx()
    ax1_t.plot(gammas, att_rmses, color="#d62728", linewidth=2, label="ATT Validation RMSE")
    ax1_t.axvline(best_gamma_att, color="#d62728", linestyle=":", label=f"Best ATT gamma*={best_gamma_att:.2f}")
    ax1_t.set_ylabel("ATT RMSE (seconds)", color="#d62728", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1_t.legend(loc="upper right", fontsize=8)

    # 2. Calibration Reliability: CBF
    ax2 = axes[0, 1]
    bins = np.linspace(0, 100, 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    dnn_binned = [np.mean(dnn_eval_cbf[(y_eval_cbf >= bins[i]) & (y_eval_cbf < bins[i+1])]) for i in range(len(bins)-1)]
    elm_binned = [np.mean(elm_eval_cbf[(y_eval_cbf >= bins[i]) & (y_eval_cbf < bins[i+1])]) for i in range(len(bins)-1)]
    ax2.plot(bin_centers, bin_centers, "k--", label="Perfect Calibration (y=x)")
    ax2.plot(bin_centers, dnn_binned, "o-", color="#1f77b4", label="Baseline DNN")
    ax2.plot(bin_centers, elm_binned, "s-", color="#2ca02c", label="Feature ELM")
    ax2.set_xlabel("True CBF (mL/100g/min)", fontweight="bold")
    ax2.set_ylabel("Predicted CBF (mL/100g/min)", fontweight="bold")
    ax2.set_title("CBF Calibration Reliability Curves", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Calibration Reliability: ATT
    ax3 = axes[1, 0]
    bins_att = np.linspace(0.5, 3.0, 26)
    bin_centers_att = 0.5 * (bins_att[:-1] + bins_att[1:])
    dnn_binned_att = [np.mean(dnn_eval_att[(y_eval_att >= bins_att[i]) & (y_eval_att < bins_att[i+1])]) for i in range(len(bins_att)-1)]
    elm_binned_att = [np.mean(elm_eval_att[(y_eval_att >= bins_att[i]) & (y_eval_att < bins_att[i+1])]) for i in range(len(bins_att)-1)]
    ax3.plot(bin_centers_att, bin_centers_att, "k--", label="Perfect Calibration (y=x)")
    ax3.plot(bin_centers_att, dnn_binned_att, "o-", color="#1f77b4", label="Baseline DNN")
    ax3.plot(bin_centers_att, elm_binned_att, "s-", color="#2ca02c", label="Feature ELM")
    ax3.set_xlabel("True ATT (seconds)", fontweight="bold")
    ax3.set_ylabel("Predicted ATT (seconds)", fontweight="bold")
    ax3.set_title("ATT Calibration Reliability Curves", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Method Comparison Bar Chart
    ax4 = axes[1, 1]
    labels_short = [
        "Base DNN",
        "Base ELM",
        "Affine DNN",
        "Affine ELM",
        "Blend g*",
        "Poly2 ELM",
        "EarlyDamped"
    ]
    rmses_cbf = df_eval["CBF_RMSE"].tolist()
    rmses_att = df_eval["ATT_RMSE"].tolist()
    x = np.arange(len(labels_short))
    w = 0.35

    ax4.bar(x - w/2, rmses_cbf, w, label="CBF RMSE (mL/100g/min)", color="#1f77b4", edgecolor="black")
    ax4.set_ylabel("CBF RMSE", color="#1f77b4", fontweight="bold")
    ax4.set_ylim(3.6, 3.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_short, rotation=30, ha="right", fontsize=8)
    ax4.set_title("Calibration Methods Comparison on Held-Out Validation", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)

    ax4_t = ax4.twinx()
    ax4_t.plot(x, rmses_att, "o-", color="#d62728", linewidth=2, label="ATT RMSE (s)")
    ax4_t.set_ylabel("ATT RMSE (s)", color="#d62728", fontweight="bold")
    ax4_t.set_ylim(0.298, 0.315)
    ax4.legend(loc="upper left", fontsize=8)
    ax4_t.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    diag_path = os.path.join(PLOTS_DIR, "stage4_calibration_diagnostics.png")
    plt.savefig(diag_path, dpi=300)
    plt.close()
    print(f"Saved calibration diagnostic plot to: {diag_path}")
    print("\nSTAGE 4 CALIBRATION EVALUATION COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()

