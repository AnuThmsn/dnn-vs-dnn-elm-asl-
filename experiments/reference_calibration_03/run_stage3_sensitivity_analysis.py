"""
run_stage3_sensitivity_analysis.py
Stage 3: Controlled Sensitivity and Robustness Analysis of Physical Reference Values
under Fixed 4-PLD Protocol.
"""

import os
import sys
import gc
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

def simulate_perturbed_signal(cbf, att, snr_arr, plds, tau, t1a, t1t, alpha, beta, lmbda, scale, rng):
    """
    Simulate ASL signal with perturbed physical parameters and dual-channel Rician noise.
    cbf: (N,), att: (N,)
    """
    N = len(cbf)
    D = len(plds)

    cbf_col = cbf[:, None]
    att_col = att[:, None]
    plds_row = plds[None, :]

    f_per_s = cbf_col / (6000.0 * lmbda)
    t_delay = np.maximum(plds_row - att_col, 0.0)
    t_end = np.maximum(tau + plds_row - att_col, 0.0)
    decay = np.exp(-t_delay / t1t) - np.exp(-t_end / t1t)
    coeff = 2.0 * alpha * beta * t1t * (1.0 / lmbda) * f_per_s * np.exp(-att_col / t1a)
    clean = coeff * decay * scale

    # Dual-channel Rician noise
    ref_f = physics.REF_CBF / (6000.0 * lmbda)
    ref_decay = np.exp(-np.maximum(physics.REF_PLD - physics.REF_ATT, 0.0) / t1t) - np.exp(-np.maximum(physics.REF_LD + physics.REF_PLD - physics.REF_ATT, 0.0) / t1t)
    ref_coeff = 2.0 * alpha * beta * t1t * (1.0 / lmbda) * ref_f * np.exp(-physics.REF_ATT / t1a)
    ref_sc = float(ref_coeff * ref_decay * scale)

    sd_arr = np.where(np.isinf(snr_arr), 0.0, ref_sc / snr_arr)[:, None]

    e1 = rng.normal(0.0, 1.0, size=(N, D)) * sd_arr
    e2 = rng.normal(0.0, 1.0, size=(N, D)) * sd_arr
    e3 = rng.normal(0.0, 1.0, size=(N, D)) * sd_arr
    e4 = rng.normal(0.0, 1.0, size=(N, D)) * sd_arr

    mc = np.sqrt((clean + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((clean + e3) ** 2 + e4 ** 2)
    return (mc + ml).astype(np.float32)


def main():
    print("=" * 80)
    print("EXPERIMENT 3 - STAGE 3: PHYSICAL REFERENCE VALUE SENSITIVITY ANALYSIS")
    print(f"Device: {device}")
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
    X_train_raw = np.array(x_tr_mmap[idx_tr], dtype=np.float32)
    Y_train_raw = np.array(y_tr_mmap[idx_tr], dtype=np.float32)
    X_train_dnn = (X_train_raw - X_mean) / X_std

    # Validation sample (50k for fast, exact sensitivity sweep)
    N_val_sub = 50_000
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')
    snr_val_mmap = np.load(os.path.join(CACHE_DIR, "SNR_val.npy"), mmap_mode='r')
    cbf_val_gt = np.array(y_val_mmap[:N_val_sub, 0], dtype=np.float64)
    att_val_gt = np.array(y_val_mmap[:N_val_sub, 1], dtype=np.float64)
    snr_val_sub = np.array(snr_val_mmap[:N_val_sub], dtype=np.float64)

    # 3. Fit Baseline Feature ELMs on Train
    print("\nFitting Baseline Feature ELMs (CBF alpha=10, ATT alpha=1000, 5 seeds)...")
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

    cbf_elms = []
    att_elms = []
    for seed in [42, 101, 2024, 7, 99]:
        elm_c = GPUChunkedELM(n_hidden=2048, alpha=10.0, activation="tanh", seed=seed)
        elm_c.fit_chunked(F_tr_cbf, y_tr_cbf_std)
        cbf_elms.append(elm_c)

        elm_a = GPUChunkedELM(n_hidden=2048, alpha=1000.0, activation="tanh", seed=seed)
        elm_a.fit_chunked(F_tr_att, y_tr_att_std)
        att_elms.append(elm_a)

    print("Baseline Feature ELMs successfully fitted.")
    del F_tr_cbf, F_tr_att
    torch.cuda.empty_cache()

    # 4. Define Parameter Sweeps & Scientific Ranges
    # Baseline defaults:
    plds_base = physics.PLDS
    tau_base = physics.TAU          # 1.8 s
    t1a_base = physics.T1A          # 1.66 s
    t1t_base = physics.T1T          # 1.20 s
    alpha_base = physics.ALPHA      # 0.85
    beta_base = physics.BETA        # 0.75
    lmbda_base = physics.LAMBDA     # 0.90 mL/g
    scale_base = physics.SCALE      # 100,000

    sensitivity_configs = [
        # Parameter Name, Param Key, Values list, Unit, Reference Source
        ("Blood_T1a", "t1a", [1.50, 1.58, 1.66, 1.74, 1.80], "s", "Lu et al. 2004; Alsop et al. 2015 consensus (1.65 +/- 0.10 s)"),
        ("Tissue_T1t", "t1t", [1.10, 1.15, 1.20, 1.25, 1.30, 1.40], "s", "Buxton et al. 1998; Wansapura et al. 1999 (1.1 - 1.4 s)"),
        ("Inversion_Efficiency_Alpha", "alpha", [0.75, 0.80, 0.85, 0.90], "dimensionless", "Alsop et al. 2015 consensus (0.75 - 0.90)"),
        ("Partition_Coefficient_Lambda", "lmbda", [0.82, 0.86, 0.90, 0.94, 0.98], "mL/g", "Herscovitch & Raichle 1985; Alsop et al. 2015 (0.82 WM - 0.98 GM)"),
        ("Signal_Scaling_M0", "scale", [80_000.0, 90_000.0, 100_000.0, 110_000.0, 120_000.0], "arb. units", "B1 receive gain / M0 calibration variation (+/- 20%)")
    ]

    results = []

    print("\n" + "=" * 80)
    print("RUNNING CONTROLLED SENSITIVITY SWEEPS")
    print("=" * 80)

    for p_name, p_key, p_vals, p_unit, p_source in sensitivity_configs:
        print(f"\n--- Testing Parameter: {p_name} ({p_unit}) ---")
        for val in p_vals:
            # Set parameters
            kwargs = {
                "plds": plds_base,
                "tau": tau_base,
                "t1a": t1a_base,
                "t1t": t1t_base,
                "alpha": alpha_base,
                "beta": beta_base,
                "lmbda": lmbda_base,
                "scale": scale_base
            }
            kwargs[p_key] = val

            sim_rng = np.random.default_rng(12345)
            x_sim = simulate_perturbed_signal(
                cbf_val_gt, att_val_gt, snr_val_sub,
                rng=sim_rng, **kwargs
            )

            # Standardize with frozen baseline statistics
            x_norm = (x_sim - X_mean) / X_std

            # 1. Baseline DNN predictions
            cbf_dnn = cbf_net.predict_physical(x_norm)
            att_dnn = att_net.predict_physical(x_norm)

            m_dnn_cbf = calc_metrics(cbf_val_gt, cbf_dnn)
            m_dnn_att = calc_metrics(att_val_gt, att_dnn)

            # 2. Extract latents for Feature ELM
            F_cbf_val = []
            F_att_val = []
            for i in range(0, N_val_sub, chunk_sz):
                F_cbf_val.append(cbf_net.penultimate_features(x_norm[i:i+chunk_sz]))
                F_att_val.append(att_net.penultimate_features(x_norm[i:i+chunk_sz]))
            F_cbf_val = np.vstack(F_cbf_val)
            F_att_val = np.vstack(F_att_val)

            cbf_elm_preds = []
            for elm in cbf_elms:
                cbf_elm_preds.append(np.clip(elm.predict(F_cbf_val) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX))
            avg_cbf_elm = np.mean(cbf_elm_preds, axis=0)
            m_elm_cbf = calc_metrics(cbf_val_gt, avg_cbf_elm)

            att_elm_preds = []
            for elm in att_elms:
                att_elm_preds.append(np.clip(elm.predict(F_att_val) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX))
            avg_att_elm = np.mean(att_elm_preds, axis=0)
            m_elm_att = calc_metrics(att_val_gt, avg_att_elm)

            del F_cbf_val, F_att_val, x_sim, x_norm
            torch.cuda.empty_cache()

            is_baseline = (
                (p_key == "t1a" and val == t1a_base) or
                (p_key == "t1t" and val == t1t_base) or
                (p_key == "alpha" and val == alpha_base) or
                (p_key == "lmbda" and val == lmbda_base) or
                (p_key == "scale" and val == scale_base)
            )

            print(f"  {p_key}={val:<8} {'[BASELINE]' if is_baseline else ''}")
            print(f"    CBF RMSE: DNN={m_dnn_cbf['RMSE']:.3f}, ELM={m_elm_cbf['RMSE']:.3f} | Bias: DNN={m_dnn_cbf['Bias']:+.2f}, ELM={m_elm_cbf['Bias']:+.2f}")
            print(f"    ATT RMSE: DNN={m_dnn_att['RMSE']:.4f}, ELM={m_elm_att['RMSE']:.4f} | Bias: DNN={m_dnn_att['Bias']:+.3f}, ELM={m_elm_att['Bias']:+.3f}")

            results.append({
                "Parameter_Name": p_name,
                "Parameter_Key": p_key,
                "Parameter_Value": val,
                "Unit": p_unit,
                "Is_Baseline": is_baseline,
                "Source_Range": p_source,
                "DNN_CBF_RMSE": m_dnn_cbf["RMSE"],
                "DNN_CBF_MAE": m_dnn_cbf["MAE"],
                "DNN_CBF_Bias": m_dnn_cbf["Bias"],
                "DNN_CBF_CCC": m_dnn_cbf["CCC"],
                "ELM_CBF_RMSE": m_elm_cbf["RMSE"],
                "ELM_CBF_MAE": m_elm_cbf["MAE"],
                "ELM_CBF_Bias": m_elm_cbf["Bias"],
                "ELM_CBF_CCC": m_elm_cbf["CCC"],
                "DNN_ATT_RMSE": m_dnn_att["RMSE"],
                "DNN_ATT_MAE": m_dnn_att["MAE"],
                "DNN_ATT_Bias": m_dnn_att["Bias"],
                "DNN_ATT_CCC": m_dnn_att["CCC"],
                "ELM_ATT_RMSE": m_elm_att["RMSE"],
                "ELM_ATT_MAE": m_elm_att["MAE"],
                "ELM_ATT_Bias": m_elm_att["Bias"],
                "ELM_ATT_CCC": m_elm_att["CCC"]
            })

    df_res = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "stage3_sensitivity_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved sensitivity results to: {csv_path}")

    # 5. Generate Sensitivity Plots
    print("\nGenerating Sensitivity Response Plots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)

    # Plot 1: Blood T1a vs CBF Bias & ATT Bias
    ax1 = axes[0, 0]
    sub1 = df_res[df_res["Parameter_Key"] == "t1a"]
    ax1.plot(sub1["Parameter_Value"], sub1["DNN_CBF_Bias"], "o-", color="#1f77b4", label="DNN CBF Bias")
    ax1.plot(sub1["Parameter_Value"], sub1["ELM_CBF_Bias"], "s--", color="#2ca02c", label="ELM CBF Bias")
    ax1.axhline(0, color="gray", linestyle=":")
    ax1.axvline(t1a_base, color="red", linestyle="--", alpha=0.6, label="Baseline (1.66 s)")
    ax1.set_xlabel("Blood T1a (seconds)", fontweight="bold")
    ax1.set_ylabel("CBF Bias (mL/100g/min)", fontweight="bold")
    ax1.set_title("Sensitivity: Blood T1a vs. CBF Bias", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Alpha vs CBF Bias
    ax2 = axes[0, 1]
    sub2 = df_res[df_res["Parameter_Key"] == "alpha"]
    ax2.plot(sub2["Parameter_Value"], sub2["DNN_CBF_Bias"], "o-", color="#1f77b4", label="DNN CBF Bias")
    ax2.plot(sub2["Parameter_Value"], sub2["ELM_CBF_Bias"], "s--", color="#2ca02c", label="ELM CBF Bias")
    ax2.axhline(0, color="gray", linestyle=":")
    ax2.axvline(alpha_base, color="red", linestyle="--", alpha=0.6, label="Baseline (0.85)")
    ax2.set_xlabel("Inversion Efficiency Alpha", fontweight="bold")
    ax2.set_ylabel("CBF Bias (mL/100g/min)", fontweight="bold")
    ax2.set_title("Sensitivity: Alpha vs. CBF Bias", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Scale (M0) vs CBF Bias
    ax3 = axes[0, 2]
    sub3 = df_res[df_res["Parameter_Key"] == "scale"]
    scale_rel = sub3["Parameter_Value"] / 100_000.0
    ax3.plot(scale_rel, sub3["DNN_CBF_Bias"], "o-", color="#1f77b4", label="DNN CBF Bias")
    ax3.plot(scale_rel, sub3["ELM_CBF_Bias"], "s--", color="#2ca02c", label="ELM CBF Bias")
    ax3.axhline(0, color="gray", linestyle=":")
    ax3.axvline(1.0, color="red", linestyle="--", alpha=0.6, label="Baseline (1.0x)")
    ax3.set_xlabel("Signal Scale Multiplier (Relative M0)", fontweight="bold")
    ax3.set_ylabel("CBF Bias (mL/100g/min)", fontweight="bold")
    ax3.set_title("Sensitivity: Scale/M0 vs. CBF Bias", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Plot 4: Blood T1a vs ATT Bias
    ax4 = axes[1, 0]
    ax4.plot(sub1["Parameter_Value"], sub1["DNN_ATT_Bias"], "o-", color="#d62728", label="DNN ATT Bias")
    ax4.plot(sub1["Parameter_Value"], sub1["ELM_ATT_Bias"], "s--", color="#9467bd", label="ELM ATT Bias")
    ax4.axhline(0, color="gray", linestyle=":")
    ax4.axvline(t1a_base, color="red", linestyle="--", alpha=0.6, label="Baseline (1.66 s)")
    ax4.set_xlabel("Blood T1a (seconds)", fontweight="bold")
    ax4.set_ylabel("ATT Bias (seconds)", fontweight="bold")
    ax4.set_title("Sensitivity: Blood T1a vs. ATT Bias", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Plot 5: Tissue T1t vs ATT Bias
    ax5 = axes[1, 1]
    sub5 = df_res[df_res["Parameter_Key"] == "t1t"]
    ax5.plot(sub5["Parameter_Value"], sub5["DNN_ATT_Bias"], "o-", color="#d62728", label="DNN ATT Bias")
    ax5.plot(sub5["Parameter_Value"], sub5["ELM_ATT_Bias"], "s--", color="#9467bd", label="ELM ATT Bias")
    ax5.axhline(0, color="gray", linestyle=":")
    ax5.axvline(t1t_base, color="red", linestyle="--", alpha=0.6, label="Baseline (1.20 s)")
    ax5.set_xlabel("Tissue T1t (seconds)", fontweight="bold")
    ax5.set_ylabel("ATT Bias (seconds)", fontweight="bold")
    ax5.set_title("Sensitivity: Tissue T1t vs. ATT Bias", fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # Plot 6: Scale (M0) vs ATT Bias
    ax6 = axes[1, 2]
    ax6.plot(scale_rel, sub3["DNN_ATT_Bias"], "o-", color="#d62728", label="DNN ATT Bias")
    ax6.plot(scale_rel, sub3["ELM_ATT_Bias"], "s--", color="#9467bd", label="ELM ATT Bias")
    ax6.axhline(0, color="gray", linestyle=":")
    ax6.axvline(1.0, color="red", linestyle="--", alpha=0.6, label="Baseline (1.0x)")
    ax6.set_xlabel("Signal Scale Multiplier (Relative M0)", fontweight="bold")
    ax6.set_ylabel("ATT Bias (seconds)", fontweight="bold")
    ax6.set_title("Sensitivity: Scale/M0 vs. ATT Bias", fontweight="bold")
    ax6.legend(fontsize=8)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "stage3_parameter_sensitivity.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {plot_path}")
    print("\nSTAGE 3 SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()

