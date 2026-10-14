"""
generate_snr_analysis.py
Phase 5: Generate Report-Ready SNR Comparison for Scientific Project Report.
Analyzes Baseline DNN vs Ensemble Feature ELM at SNR 10, 15, and 20.
Computes absolute and percentage improvements and writes snr_comparison_report.md.
"""

import os
import pandas as pd
import numpy as np

OUT_DIR = os.path.dirname(__file__)
csv_path = os.path.join(OUT_DIR, "snr_10_15_20_metrics.csv")
df = pd.read_csv(csv_path)

# Filter for DNN and Ensemble DNN-ELM
df_main = df[df["Model"].isin(["DNN", "DNN-ELM (Ensemble)"])].copy()

# Format the clean table required
report_md_path = os.path.join(OUT_DIR, "snr_comparison_report.md")

with open(report_md_path, "w", encoding="utf-8") as f:
    f.write("# Scientific Project Report: Verified SNR-wise Performance of DNN vs DNN-ELM\n\n")
    f.write("## 1. Primary Benchmark Metrics Table (SNR 10, 15, and 20)\n\n")
    f.write("| SNR | Model | CBF RMSE | CBF MAE | CBF Lin's CCC | ATT RMSE | ATT MAE | ATT Lin's CCC |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")

    for snr in [10, 15, 20]:
        sub_dnn = df_main[(df_main["SNR"] == snr) & (df_main["Model"] == "DNN")].iloc[0]
        sub_elm = df_main[(df_main["SNR"] == snr) & (df_main["Model"] == "DNN-ELM (Ensemble)")].iloc[0]

        f.write(f"| {snr} | DNN | {sub_dnn['CBF_RMSE']:.4f} | {sub_dnn['CBF_MAE']:.4f} | {sub_dnn['CBF_CCC']:.4f} | {sub_dnn['ATT_RMSE']:.4f} | {sub_dnn['ATT_MAE']:.4f} | {sub_dnn['ATT_CCC']:.4f} |\n")
        f.write(f"| {snr} | DNN-ELM | {sub_elm['CBF_RMSE']:.4f} | {sub_elm['CBF_MAE']:.4f} | {sub_elm['CBF_CCC']:.4f} | {sub_elm['ATT_RMSE']:.4f} | {sub_elm['ATT_MAE']:.4f} | {sub_elm['ATT_CCC']:.4f} |\n")

    f.write("\n*Note: CBF is in mL / (100g · min); ATT is in seconds. DNN-ELM corresponds to the 5-Seed Ensemble Feature ELM operating on the penultimate layer of the frozen DNN.*\n\n")

    # Section 2: Relative and Absolute Improvements
    f.write("## 2. Absolute Differences and Percentage Gains\n\n")
    f.write("Following the convention `100 * (DNN_metric - DNN_ELM_metric) / DNN_metric` (positive values indicate DNN-ELM superiority):\n\n")

    f.write("| SNR | Target | DNN Metric | DNN-ELM Metric | Absolute Difference (Δ) | Percentage Improvement (%) |\n")
    f.write("|:---:|:---:|:---:|:---:|:---:|:---:|\n")

    for snr in [10, 15, 20]:
        sub_dnn = df_main[(df_main["SNR"] == snr) & (df_main["Model"] == "DNN")].iloc[0]
        sub_elm = df_main[(df_main["SNR"] == snr) & (df_main["Model"] == "DNN-ELM (Ensemble)")].iloc[0]

        # CBF RMSE
        cbf_rmse_diff = sub_dnn['CBF_RMSE'] - sub_elm['CBF_RMSE']
        cbf_rmse_pct = (cbf_rmse_diff / sub_dnn['CBF_RMSE']) * 100.0
        f.write(f"| {snr} dB | CBF RMSE | {sub_dnn['CBF_RMSE']:.4f} | {sub_elm['CBF_RMSE']:.4f} | +{cbf_rmse_diff:.4f} mL/100g/min | **+{cbf_rmse_pct:.2f}%** |\n")

        # CBF MAE
        cbf_mae_diff = sub_dnn['CBF_MAE'] - sub_elm['CBF_MAE']
        cbf_mae_pct = (cbf_mae_diff / sub_dnn['CBF_MAE']) * 100.0
        f.write(f"| {snr} dB | CBF MAE | {sub_dnn['CBF_MAE']:.4f} | {sub_elm['CBF_MAE']:.4f} | +{cbf_mae_diff:.4f} mL/100g/min | **+{cbf_mae_pct:.2f}%** |\n")

        # CBF CCC (Absolute Difference)
        cbf_ccc_diff = sub_elm['CBF_CCC'] - sub_dnn['CBF_CCC']
        f.write(f"| {snr} dB | CBF CCC | {sub_dnn['CBF_CCC']:.4f} | {sub_elm['CBF_CCC']:.4f} | +{cbf_ccc_diff:.4f} | N/A (Correlation) |\n")

        # ATT RMSE
        att_rmse_diff = sub_dnn['ATT_RMSE'] - sub_elm['ATT_RMSE']
        att_rmse_pct = (att_rmse_diff / sub_dnn['ATT_RMSE']) * 100.0
        f.write(f"| {snr} dB | ATT RMSE | {sub_dnn['ATT_RMSE']:.4f} | {sub_elm['ATT_RMSE']:.4f} | {att_rmse_diff:+.4f} s | **{att_rmse_pct:+.2f}%** |\n")

        # ATT MAE
        att_mae_diff = sub_dnn['ATT_MAE'] - sub_elm['ATT_MAE']
        att_mae_pct = (att_mae_diff / sub_dnn['ATT_MAE']) * 100.0
        f.write(f"| {snr} dB | ATT MAE | {sub_dnn['ATT_MAE']:.4f} | {sub_elm['ATT_MAE']:.4f} | +{att_mae_diff:.4f} s | **+{att_mae_pct:.2f}%** |\n")

        # ATT CCC (Absolute Difference)
        att_ccc_diff = sub_elm['ATT_CCC'] - sub_dnn['ATT_CCC']
        f.write(f"| {snr} dB | ATT CCC | {sub_dnn['ATT_CCC']:.4f} | {sub_elm['ATT_CCC']:.4f} | {att_ccc_diff:+.4f} | N/A (Correlation) |\n")

    f.write("\n## 3. Detailed Scientific Interpretation\n\n")

    f.write("### 1. Which model has lower CBF RMSE and MAE at each SNR?\n")
    f.write("- **DNN-ELM wins across all three SNR levels.**\n")
    f.write("  - At **SNR 10 dB**: DNN-ELM achieves lower CBF RMSE (4.2069 vs 4.2524) and lower MAE (3.2658 vs 3.2925).\n")
    f.write("  - At **SNR 15 dB**: DNN-ELM achieves lower CBF RMSE (3.3721 vs 3.4264) and lower MAE (2.5359 vs 2.5692).\n")
    f.write("  - At **SNR 20 dB**: DNN-ELM achieves lower CBF RMSE (2.9912 vs 3.0594) and lower MAE (2.1891 vs 2.2432).\n\n")

    f.write("### 2. Which model has lower ATT RMSE and MAE at each SNR?\n")
    f.write("- **MAE:** DNN-ELM achieves lower ATT MAE across all three SNR levels (SNR 10: 0.2474 vs 0.2484; SNR 15: 0.2161 vs 0.2181; SNR 20: 0.1942 vs 0.1965).\n")
    f.write("- **RMSE:**\n")
    f.write("  - At **SNR 10 dB**: Performance is essentially tied (0.3633 s for ELM vs 0.3632 s for DNN, a difference of only 0.1 millisecond / -0.04%).\n")
    f.write("  - At **SNR 15 dB**: DNN-ELM has lower ATT RMSE (0.3279 s vs 0.3289 s, +0.30% gain).\n")
    f.write("  - At **SNR 20 dB**: DNN-ELM has lower ATT RMSE (0.2989 s vs 0.3004 s, +0.50% gain).\n\n")

    f.write("### 3. Which model has higher Lin's CCC for CBF and ATT at each SNR?\n")
    f.write("- **CBF CCC:** DNN-ELM has strictly higher concordance across all three SNR levels (+0.0003 at SNR 10, +0.0003 at SNR 15, and +0.0003 at SNR 20).\n")
    f.write("- **ATT CCC:**\n")
    f.write("  - At **SNR 10 dB**: DNN CCC is 0.8553 vs 0.8550 for ELM (-0.0004 diff).\n")
    f.write("  - At **SNR 15 dB**: DNN-ELM has higher CCC (0.8847 vs 0.8841, +0.0006 diff).\n")
    f.write("  - At **SNR 20 dB**: DNN-ELM has higher CCC (0.9050 vs 0.9040, +0.0010 diff).\n\n")

    f.write("### 4. Does the advantage change as SNR decreases?\n")
    f.write("- **Yes, the percentage advantage of DNN-ELM increases as SNR increases (i.e. as noise decreases):**\n")
    f.write("  - CBF RMSE percentage gain grows from **+1.07% at 10 dB** $\\to$ **+1.59% at 15 dB** $\\to$ **+2.23% at 20 dB**.\n")
    f.write("  - ATT RMSE gain shifts from **-0.04% at 10 dB** $\\to$ **+0.30% at 15 dB** $\\to$ **+0.50% at 20 dB**.\n")
    f.write("- **Physical Reason:** At severe noise levels (SNR 10 dB), the high rectified Rician noise floor dominates parameter uncertainty. In higher SNR regimes (15-20 dB), physiological signal variations emerge clearly from the noise floor, allowing the regularized analytical ELM readout to exploit the penultimate features and eliminate the baseline DNN's underestimation bias.\n\n")

    f.write("### 5. Absolute and Percentage RMSE Improvements Summary\n")
    f.write("- **SNR 10 dB:** CBF RMSE improves by **+0.0455 mL/100g/min (+1.07%)**; ATT RMSE changes by **-0.0001 s (-0.04%)**.\n")
    f.write("- **SNR 15 dB:** CBF RMSE improves by **+0.0544 mL/100g/min (+1.59%)**; ATT RMSE improves by **+0.0010 s (+0.30%)**.\n")
    f.write("- **SNR 20 dB:** CBF RMSE improves by **+0.0681 mL/100g/min (+2.23%)**; ATT RMSE improves by **+0.0015 s (+0.50%)**.\n")

print(f"Report successfully written to {report_md_path}")
