"""
phase5_sensitivity.py
Phase 5: Analytical Sensitivity and Physical Identifiability Analysis.
Evaluates the Jacobian, Fisher Information Matrix, condition numbers,
and gradient collinearity of the Buxton pCASL model across ATT in [0.5, 3.0] s.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
OUT_DIR = os.path.join(EXP_DIR, "significant_improvement_20261012")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

sys.path.append(EXP_DIR)
import physics

def compute_jacobian(cbf_val, att_val, eps=1e-5):
    """Numerical Jacobian [dS/dcbf, dS/datt] across 4 PLDs."""
    # Base signal (4,)
    s0 = physics.compute_physical_signals(physics.PLDS, physics.TAU, cbf_val, att_val).squeeze()

    # Partial w.r.t CBF
    s_cbf_plus = physics.compute_physical_signals(physics.PLDS, physics.TAU, cbf_val + eps, att_val).squeeze()
    ds_dcbf = (s_cbf_plus - s0) / eps

    # Partial w.r.t ATT
    s_att_plus = physics.compute_physical_signals(physics.PLDS, physics.TAU, cbf_val, att_val + eps).squeeze()
    ds_datt = (s_att_plus - s0) / eps

    # Normalize columns for scale-independent collinearity
    norm_cbf = np.linalg.norm(ds_dcbf)
    norm_att = np.linalg.norm(ds_datt)
    cos_sim = np.dot(ds_dcbf, ds_datt) / (norm_cbf * norm_att) if (norm_cbf > 0 and norm_att > 0) else 0.0

    # 4x2 Jacobian
    J = np.stack([ds_dcbf, ds_datt], axis=1)
    # Singular values
    sv = np.linalg.svd(J, compute_uv=False)
    cond_num = sv[0] / (sv[1] + 1e-15)

    return {
        "ds_dcbf": ds_dcbf,
        "ds_datt": ds_datt,
        "norm_cbf_grad": float(norm_cbf),
        "norm_att_grad": float(norm_att),
        "cosine_collinearity": float(cos_sim),
        "cond_num": float(cond_num),
        "min_singular_val": float(sv[1]),
        "max_singular_val": float(sv[0])
    }

def main():
    print("=" * 80)
    print("PHASE 5: SENSITIVITY AND PHYSICAL IDENTIFIABILITY ANALYSIS")
    print("=" * 80)

    att_range = np.linspace(0.5, 3.0, 101)
    cbf_eval_levels = [30.0, 50.0, 80.0]

    sensitivity_rows = []
    for cbf in cbf_eval_levels:
        for att in att_range:
            res = compute_jacobian(cbf, att)
            sensitivity_rows.append({
                "CBF": cbf,
                "ATT": att,
                "Cosine_Collinearity": res["cosine_collinearity"],
                "Condition_Number": res["cond_num"],
                "Min_Singular_Value": res["min_singular_val"],
                "Max_Singular_Value": res["max_singular_val"],
                "Norm_ATT_Grad": res["norm_att_grad"],
                "Norm_CBF_Grad": res["norm_cbf_grad"]
            })

    df_sens = pd.DataFrame(sensitivity_rows)
    csv_path = os.path.join(TABLES_DIR, "att_identifiability_metrics.csv")
    df_sens.to_csv(csv_path, index=False)
    print(f"Saved identifiability metrics to: {csv_path}")

    # Summary table for predefined ATT intervals at nominal CBF=50
    sub_50 = df_sens[df_sens["CBF"] == 50.0]
    intervals = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]
    int_summary = []
    for lo, hi in intervals:
        m = sub_50[(sub_50["ATT"] >= lo) & (sub_50["ATT"] < hi)] if hi < 3.0 else sub_50[(sub_50["ATT"] >= lo) & (sub_50["ATT"] <= hi)]
        int_summary.append({
            "ATT_Interval": f"[{lo:.1f}, {hi:.1f}]",
            "Mean_Collinearity": m["Cosine_Collinearity"].mean(),
            "Max_Collinearity": m["Cosine_Collinearity"].max(),
            "Mean_Condition_Number": m["Condition_Number"].mean(),
            "Mean_ATT_Grad_Magnitude": m["Norm_ATT_Grad"].mean()
        })
    df_int = pd.DataFrame(int_summary)
    print("\nPhysical Identifiability Breakdown by ATT Interval (Nominal CBF=50):")
    print(df_int.to_string(index=False))

    # Generate Diagnostics Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Plot 1: Cosine Collinearity vs ATT
    for cbf in cbf_eval_levels:
        sub = df_sens[df_sens["CBF"] == cbf]
        axes[0].plot(sub["ATT"], np.abs(sub["Cosine_Collinearity"]), lw=2.2, label=f"CBF = {int(cbf)} mL")
    axes[0].axvspan(0.5, 1.0, color="red", alpha=0.15, label="Early Arrival Window [0.5, 1.0)s")
    axes[0].axvline(1.525, color="black", ls="--", alpha=0.7, label="First Delay (PLD1 = 1.525s)")
    axes[0].set_xlabel("Arterial Transit Time (ATT) [seconds]", fontsize=11)
    axes[0].set_ylabel("|Cosine Collinearity| between dS/dCBF and dS/dATT", fontsize=11)
    axes[0].set_title("Parameter Gradient Collinearity (|cos θ|)", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Plot 2: Condition Number of Jacobian vs ATT
    for cbf in cbf_eval_levels:
        sub = df_sens[df_sens["CBF"] == cbf]
        axes[1].plot(sub["ATT"], np.log10(sub["Condition_Number"]), lw=2.2, label=f"CBF = {int(cbf)} mL")
    axes[1].axvspan(0.5, 1.0, color="red", alpha=0.15, label="Early Arrival Window [0.5, 1.0)s")
    axes[1].axvline(1.525, color="black", ls="--", alpha=0.7, label="First Delay (PLD1 = 1.525s)")
    axes[1].set_xlabel("Arterial Transit Time (ATT) [seconds]", fontsize=11)
    axes[1].set_ylabel("log10(Condition Number)", fontsize=11)
    axes[1].set_title("Inverse Problem Ill-Conditioning (log10 κ)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "phase5_att_identifiability_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved diagnostic plots to: {plot_path}")
    print("=" * 80)
    print("PHASE 5 SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

