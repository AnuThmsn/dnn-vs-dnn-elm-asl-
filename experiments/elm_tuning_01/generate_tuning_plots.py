"""
generate_tuning_plots.py
Stage 4: Generate comprehensive diagnostic plots for ELM hyperparameter optimization.
Plots:
1. val_rmse_vs_alpha.png: Validation RMSE vs. alpha for each hidden neuron count (curves for 128, 256, 512, 1024, 2048) for CBF and ATT.
2. hidden_neurons_vs_metrics.png: Hidden neuron count vs. validation RMSE/MAE for different activations.
3. seed_variability_boxplots.png: Box plots and distributions across 10 random seeds for top candidates vs. baseline.
4. pareto_tradeoff_capacity_time.png: Validation RMSE vs. Fit Time across architectures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = r"D:\dnn vs dnn elm\experiments\elm_tuning_01"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_rmse_vs_alpha(df_sweep):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    hidden_colors = {
        128: '#1f77b4',
        256: '#ff7f0e',
        512: '#2ca02c',
        1024: '#d62728',
        2048: '#9467bd'
    }

    # CBF Subplot
    ax_cbf = axes[0]
    df_cbf_tanh = df_sweep[(df_sweep["Target"] == "CBF") & (df_sweep["Activation"] == "tanh")]
    for nh in sorted(df_cbf_tanh["Hidden_Neurons"].unique()):
        sub = df_cbf_tanh[df_cbf_tanh["Hidden_Neurons"] == nh].sort_values("Alpha")
        ax_cbf.plot(sub["Alpha"], sub["Val_RMSE"], marker='o', linewidth=2, markersize=5,
                    label=f"H = {nh}", color=hidden_colors.get(nh, 'gray'))

    ax_cbf.set_xscale("log")
    ax_cbf.set_xlabel(r"Regularization Parameter $\alpha$ (log scale)", fontsize=11, fontweight='bold')
    ax_cbf.set_ylabel(r"Validation CBF RMSE (mL/100g/min)", fontsize=11, fontweight='bold')
    ax_cbf.set_title("CBF Validation RMSE vs. Regularization", fontsize=12, fontweight='bold', pad=10)
    ax_cbf.legend(title="Hidden Neurons", frameon=True, fontsize=9)
    ax_cbf.grid(True, linestyle="--", alpha=0.6)

    # ATT Subplot
    ax_att = axes[1]
    df_att_tanh = df_sweep[(df_sweep["Target"] == "ATT") & (df_sweep["Activation"] == "tanh")]
    for nh in sorted(df_att_tanh["Hidden_Neurons"].unique()):
        sub = df_att_tanh[df_att_tanh["Hidden_Neurons"] == nh].sort_values("Alpha")
        ax_att.plot(sub["Alpha"], sub["Val_RMSE"], marker='s', linewidth=2, markersize=5,
                    label=f"H = {nh}", color=hidden_colors.get(nh, 'gray'))

    ax_att.set_xscale("log")
    ax_att.set_xlabel(r"Regularization Parameter $\alpha$ (log scale)", fontsize=11, fontweight='bold')
    ax_att.set_ylabel(r"Validation ATT RMSE (s)", fontsize=11, fontweight='bold')
    ax_att.set_title("ATT Validation RMSE vs. Regularization", fontsize=12, fontweight='bold', pad=10)
    ax_att.legend(title="Hidden Neurons", frameon=True, fontsize=9)
    ax_att.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "val_rmse_vs_alpha.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def plot_hidden_neurons_vs_metrics(df_sweep):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # CBF
    ax_cbf = axes[0]
    # For each activation, find the best alpha per hidden neuron count
    df_cbf = df_sweep[df_sweep["Target"] == "CBF"]
    act_colors = {"tanh": "#1f77b4", "elu": "#2ca02c", "relu": "#d62728"}
    act_markers = {"tanh": "o", "elu": "^", "relu": "s"}

    for act in df_cbf["Activation"].unique():
        sub_act = df_cbf[df_cbf["Activation"] == act]
        best_per_h = sub_act.groupby("Hidden_Neurons").apply(lambda g: g.loc[g["Val_RMSE"].idxmin()]).reset_index(drop=True)
        best_per_h = best_per_h.sort_values("Hidden_Neurons")
        ax_cbf.plot(best_per_h["Hidden_Neurons"], best_per_h["Val_RMSE"],
                    marker=act_markers.get(act, 'o'), linewidth=2, markersize=6,
                    label=f"Act: {act}", color=act_colors.get(act, 'black'))

    ax_cbf.set_xlabel("Hidden Neuron Count (H)", fontsize=11, fontweight='bold')
    ax_cbf.set_ylabel("Best Validation CBF RMSE (mL/100g/min)", fontsize=11, fontweight='bold')
    ax_cbf.set_title("CBF: Capacity vs. Validation RMSE by Activation", fontsize=12, fontweight='bold', pad=10)
    ax_cbf.set_xticks([128, 256, 512, 1024, 2048])
    ax_cbf.legend(frameon=True, fontsize=9)
    ax_cbf.grid(True, linestyle="--", alpha=0.6)

    # ATT
    ax_att = axes[1]
    df_att = df_sweep[df_sweep["Target"] == "ATT"]
    for act in df_att["Activation"].unique():
        sub_act = df_att[df_att["Activation"] == act]
        best_per_h = sub_act.groupby("Hidden_Neurons").apply(lambda g: g.loc[g["Val_RMSE"].idxmin()]).reset_index(drop=True)
        best_per_h = best_per_h.sort_values("Hidden_Neurons")
        ax_att.plot(best_per_h["Hidden_Neurons"], best_per_h["Val_RMSE"],
                    marker=act_markers.get(act, 's'), linewidth=2, markersize=6,
                    label=f"Act: {act}", color=act_colors.get(act, 'black'))

    ax_att.set_xlabel("Hidden Neuron Count (H)", fontsize=11, fontweight='bold')
    ax_att.set_ylabel("Best Validation ATT RMSE (s)", fontsize=11, fontweight='bold')
    ax_att.set_title("ATT: Capacity vs. Validation RMSE by Activation", fontsize=12, fontweight='bold', pad=10)
    ax_att.set_xticks([128, 256, 512, 1024, 2048])
    ax_att.legend(frameon=True, fontsize=9)
    ax_att.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "hidden_neurons_vs_metrics.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def plot_seed_variability(df_seeds):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Filter out ensemble rows for boxplots
    numeric_seeds = df_seeds[df_seeds["Seed"].apply(lambda s: str(s).isdigit())]

    # CBF
    ax_cbf = axes[0]
    df_cbf_seeds = numeric_seeds[numeric_seeds["Target"] == "CBF"]
    configs_cbf = df_cbf_seeds["Configuration_Name"].unique()
    data_cbf = [df_cbf_seeds[df_cbf_seeds["Configuration_Name"] == cfg]["Val_RMSE"].values for cfg in configs_cbf]

    box_cbf = ax_cbf.boxplot(data_cbf, tick_labels=configs_cbf, patch_artist=True,
                             boxprops=dict(facecolor='#a6cee3', color='#1f78b4'),
                             medianprops=dict(color='#e31a1c', linewidth=2))

    # Add ensemble horizontal lines if available
    ens5_cbf = df_seeds[(df_seeds["Target"] == "CBF") & (df_seeds["Seed"] == "Ensemble_5")]
    for _, row in ens5_cbf.iterrows():
        cfg_name = row["Configuration_Name"].replace("_5Seed_Ensemble", "")
        if cfg_name in list(configs_cbf):
            idx = list(configs_cbf).index(cfg_name) + 1
            ax_cbf.scatter(idx, row["Val_RMSE"], color="darkgreen", marker="D", s=70, zorder=5, label=f"5-Seed Ensemble" if idx==1 else "")

    ax_cbf.set_ylabel("Validation CBF RMSE (mL/100g/min)", fontsize=11, fontweight='bold')
    ax_cbf.set_title("CBF: 10-Seed Variability (Seeds 42-51)", fontsize=12, fontweight='bold', pad=10)
    ax_cbf.grid(True, linestyle="--", alpha=0.6)
    ax_cbf.tick_params(axis='x', rotation=15)
    ax_cbf.legend(frameon=True, fontsize=9)

    # ATT
    ax_att = axes[1]
    df_att_seeds = numeric_seeds[numeric_seeds["Target"] == "ATT"]
    configs_att = df_att_seeds["Configuration_Name"].unique()
    data_att = [df_att_seeds[df_att_seeds["Configuration_Name"] == cfg]["Val_RMSE"].values for cfg in configs_att]

    box_att = ax_att.boxplot(data_att, tick_labels=configs_att, patch_artist=True,
                             boxprops=dict(facecolor='#b2df8a', color='#33a02c'),
                             medianprops=dict(color='#e31a1c', linewidth=2))

    ens5_att = df_seeds[(df_seeds["Target"] == "ATT") & (df_seeds["Seed"] == "Ensemble_5")]
    for _, row in ens5_att.iterrows():
        cfg_name = row["Configuration_Name"].replace("_5Seed_Ensemble", "")
        if cfg_name in list(configs_att):
            idx = list(configs_att).index(cfg_name) + 1
            ax_att.scatter(idx, row["Val_RMSE"], color="darkgreen", marker="D", s=70, zorder=5, label=f"5-Seed Ensemble" if idx==1 else "")

    ax_att.set_ylabel("Validation ATT RMSE (s)", fontsize=11, fontweight='bold')
    ax_att.set_title("ATT: 10-Seed Variability (Seeds 42-51)", fontsize=12, fontweight='bold', pad=10)
    ax_att.grid(True, linestyle="--", alpha=0.6)
    ax_att.tick_params(axis='x', rotation=15)
    ax_att.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "seed_variability_boxplots.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def plot_test_snr_breakdown():
    snr_csv = os.path.join(OUT_DIR, "snr_stratified_test_metrics.csv")
    if not os.path.exists(snr_csv):
        return
    df_snr = pd.read_csv(snr_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    model_colors = {
        "Baseline A (DNN-G)": "#1f77b4",
        "Historical Feature ELM (5 seeds)": "#ff7f0e",
        "Tuned ELM Single Best": "#2ca02c",
        "Tuned ELM Ensemble (5 seeds)": "#d62728",
        "Tuned ELM Ensemble (10 seeds)": "#9467bd"
    }

    # Plot CBF RMSE vs SNR
    ax_cbf = axes[0]
    models = df_snr["Model"].unique()
    for m in models:
        sub = df_snr[df_snr["Model"] == m]
        x_vals = range(len(sub))
        ax_cbf.plot(x_vals, sub["CBF_RMSE"], marker='o', linewidth=2, markersize=5,
                    label=m, color=model_colors.get(m, 'gray'))
    sub0 = df_snr[df_snr["Model"] == models[0]]
    ax_cbf.set_xticks(range(len(sub0)))
    ax_cbf.set_xticklabels(sub0["SNR"], rotation=30)
    ax_cbf.set_xlabel("SNR Level", fontsize=11, fontweight='bold')
    ax_cbf.set_ylabel("CBF RMSE (mL/100g/min)", fontsize=11, fontweight='bold')
    ax_cbf.set_title("Locked Test CBF RMSE across SNR Levels", fontsize=12, fontweight='bold')
    ax_cbf.legend(fontsize=8, frameon=True)
    ax_cbf.grid(True, linestyle="--", alpha=0.6)

    # Plot ATT RMSE vs SNR
    ax_att = axes[1]
    for m in models:
        sub = df_snr[df_snr["Model"] == m]
        x_vals = range(len(sub))
        ax_att.plot(x_vals, sub["ATT_RMSE"], marker='s', linewidth=2, markersize=5,
                    label=m, color=model_colors.get(m, 'gray'))
    ax_att.set_xticks(range(len(sub0)))
    ax_att.set_xticklabels(sub0["SNR"], rotation=30)
    ax_att.set_xlabel("SNR Level", fontsize=11, fontweight='bold')
    ax_att.set_ylabel("ATT RMSE (s)", fontsize=11, fontweight='bold')
    ax_att.set_title("Locked Test ATT RMSE across SNR Levels", fontsize=12, fontweight='bold')
    ax_att.legend(fontsize=8, frameon=True)
    ax_att.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "test_evaluation_summary.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def main():
    sweep_csv = os.path.join(OUT_DIR, "validation_sweep.csv")
    seeds_csv = os.path.join(OUT_DIR, "seed_variability.csv")

    if not os.path.exists(sweep_csv) or not os.path.exists(seeds_csv):
        print("Data files not found yet. Awaiting sweep completion.")
        return

    df_sweep = pd.read_csv(sweep_csv)
    df_seeds = pd.read_csv(seeds_csv)

    print(f"Loaded {len(df_sweep)} sweep records and {len(df_seeds)} seed records.")
    plot_rmse_vs_alpha(df_sweep)
    plot_hidden_neurons_vs_metrics(df_sweep)
    plot_seed_variability(df_seeds)
    plot_test_snr_breakdown()
    print("All diagnostic plots generated successfully.")

if __name__ == "__main__":
    main()

