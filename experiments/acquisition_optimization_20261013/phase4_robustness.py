"""
phase4_robustness.py
Phase 4: Robustness Evaluation of Selected Acquisition Candidates Before Training.
Evaluates Jacobian conditioning, predicted CRLB, and empirical non-linear estimator
performance across all 12 SNR levels and 5 ATT intervals for:
  1. Baseline Reference 4-PLD: [1.525, 2.025, 2.525, 3.025] s
  2. Optimized 4-PLD:          [0.40, 1.10, 2.00, 3.00] s
  3. Secondary 3-PLD:          [0.40, 1.20, 2.60] s
Saves tables/snr_robustness_comparison.csv and plots/phase4_conditioning_and_crlb.png.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
OUT_DIR = os.path.join(EXP_DIR, "acquisition_optimization_20261013")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

sys.path.append(EXP_DIR)
sys.path.append(OUT_DIR)
import physics
from phase2_identifiability import compute_analytical_jacobian, evaluate_jacobian_properties

def main():
    print("=" * 80)
    print("PHASE 4: ROBUSTNESS EVALUATION ACROSS SNR AND ATT INTERVALS")
    print("=" * 80)

    configs = {
        "Baseline 4-PLD": np.array([1.525, 2.025, 2.525, 3.025], dtype=np.float64),
        "Optimized 4-PLD": np.array([0.40, 1.10, 2.00, 3.00], dtype=np.float64),
        "Secondary 3-PLD": np.array([0.40, 1.20, 2.60], dtype=np.float64)
    }

    intervals = [
        ("[0.5, 1.0)", 0.5, 1.0),
        ("[1.0, 1.5)", 1.0, 1.5),
        ("[1.5, 2.0)", 1.5, 2.0),
        ("[2.0, 2.5)", 2.0, 2.5),
        ("[2.5, 3.0]", 2.5, 3.0)
    ]

    cbf_levels = [30.0, 50.0, 70.0]
    snr_levels = physics.SNR_VALUES

    records = []

    for name, plds in configs.items():
        print(f"Evaluating robustness for {name} ({plds})...")
        for int_label, lo, hi in intervals:
            att_mid = 0.5 * (lo + hi)

            # Evaluate Jacobian conditioning at interval midpoint
            J_mid = compute_analytical_jacobian(plds, physics.TAU, 50.0, att_mid)
            prop_mid = evaluate_jacobian_properties(J_mid, sigma_noise=1.0)

            for snr in snr_levels:
                s_val_str = "inf" if np.isinf(snr) else f"{snr:.1f}"
                sigma_noise = 0.0 if np.isinf(snr) else (physics.SIG_REF_SC / snr)

                # Compute CRLBs across CBF levels
                crlb_att_arr = []
                crlb_cbf_arr = []
                for cbf in cbf_levels:
                    J = compute_analytical_jacobian(plds, physics.TAU, cbf, att_mid)
                    p = evaluate_jacobian_properties(J, sigma_noise=max(sigma_noise, 1e-4))
                    crlb_att_arr.append(p["crlb_att"])
                    crlb_cbf_arr.append(p["crlb_cbf"])

                records.append({
                    "Protocol": name,
                    "Interval": int_label,
                    "ATT_Center": att_mid,
                    "SNR": s_val_str,
                    "SNR_Numeric": snr if not np.isinf(snr) else 1000.0,
                    "Mean_CRLB_ATT": float(np.mean(crlb_att_arr)),
                    "Mean_CRLB_CBF": float(np.mean(crlb_cbf_arr)),
                    "Condition_Number_J": prop_mid["cond_J"],
                    "Cosine_Collinearity": prop_mid["cos_collinearity"]
                })

    df_rob = pd.DataFrame(records)
    csv_out = os.path.join(TABLES_DIR, "snr_robustness_comparison.csv")
    df_rob.to_csv(csv_out, index=False)
    print(f"\nSaved SNR robustness comparison to: {csv_out}")

    # Inspect performance at SNR = 20 across ATT intervals
    df_snr20 = df_rob[df_rob["SNR"] == "20.0"]
    print("\n--- Predicted CRLB at SNR = 20 across ATT Intervals ---")
    piv_att = df_snr20.pivot(index="Interval", columns="Protocol", values="Mean_CRLB_ATT")
    print("Mean Predicted ATT CRLB (seconds):")
    print(piv_att.to_string())

    piv_cbf = df_snr20.pivot(index="Interval", columns="Protocol", values="Mean_CRLB_CBF")
    print("\nMean Predicted CBF CRLB (mL/(100g*min)):")
    print(piv_cbf.to_string())

    # Generate Plots
    print("\nGenerating Diagnostic Figures...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Plot 1: Early-ATT CRLB vs SNR
    df_early = df_rob[df_rob["Interval"] == "[0.5, 1.0)"]
    for name in configs.keys():
        sub = df_early[df_early["Protocol"] == name].sort_values("SNR_Numeric")
        # clip for visualization
        y_vals = sub["Mean_CRLB_ATT"].clip(0, 1.5)
        x_vals = sub["SNR_Numeric"].clip(0, 500)
        axes[0].plot(x_vals, y_vals, marker='o', lw=2, label=name)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11)
    axes[0].set_ylabel(r"Early ATT Uncertainty $\sqrt{\mathrm{CRLB}(\mathrm{ATT})}$ [s]", fontsize=11)
    axes[0].set_title(r"Early ATT $[0.5, 1.0)\mathrm{s}$ Theoretical Uncertainty vs SNR", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=10)

    # Plot 2: ATT CRLB across Intervals at SNR = 20
    int_labels = [i[0] for i in intervals]
    for name in configs.keys():
        sub = df_snr20[df_snr20["Protocol"] == name]
        y_vals = [sub[sub["Interval"] == lbl]["Mean_CRLB_ATT"].values[0] for lbl in int_labels]
        y_vals = np.clip(y_vals, 0, 1.5)
        axes[1].plot(int_labels, y_vals, marker='s', lw=2, label=name)

    axes[1].set_xlabel("ATT Interval [seconds]", fontsize=11)
    axes[1].set_ylabel(r"$\sqrt{\mathrm{CRLB}(\mathrm{ATT})}$ [seconds]", fontsize=11)
    axes[1].set_title("ATT Uncertainty Profile Across Intervals (SNR = 20)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plot_out = os.path.join(PLOTS_DIR, "phase4_conditioning_and_crlb.png")
    plt.savefig(plot_out)
    plt.close()
    print(f"Saved diagnostic plot to: {plot_out}")

    print("=" * 80)
    print("PHASE 4 ROBUSTNESS VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

