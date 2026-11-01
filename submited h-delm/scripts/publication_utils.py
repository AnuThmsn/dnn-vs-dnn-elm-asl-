"""Shared publication outputs built only from the packaged experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
FIG_DIRS = {kind: ROOT / "figures" / kind for kind in ("png", "pdf", "svg")}
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
DATASETS = ROOT / "datasets"
PREDICTIONS_PATH = RESULTS / "predictions.npz"
SEED_BOOTSTRAP = 20261102
N_BOOTSTRAP = 5000

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})


def load_predictions():
    return np.load(PREDICTIONS_PATH)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    denom = var_true + var_pred + (np.mean(y_true) - np.mean(y_pred)) ** 2
    ccc = 2 * np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))) / denom
    mse = np.mean(error**2)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "Lin_CCC": float(ccc),
    }


def bootstrap_rmse_improvement(y_true, dnn, hdelm, rng, n_bootstrap=N_BOOTSTRAP):
    """Paired percentile CI for RMSE_DNN - RMSE_HDELM using resampled observations."""
    n = len(y_true)
    improvements = np.empty(n_bootstrap, dtype=np.float64)
    batch = 100
    for start in range(0, n_bootstrap, batch):
        stop = min(start + batch, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        y = y_true[indices]
        dnn_error = dnn[indices] - y
        hdelm_error = hdelm[indices] - y
        improvements[start:stop] = np.sqrt(np.mean(dnn_error**2, axis=1)) - np.sqrt(
            np.mean(hdelm_error**2, axis=1)
        )
    return tuple(np.quantile(improvements, [0.025, 0.975]).tolist())


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted


def prepare_tables():
    pred = load_predictions()
    truth = pred["Y_test"].astype(np.float64)
    snr = pred["snr_test"]
    levels = (10.0, 15.0, 20.0)
    parameters = ("CBF", "ATT")
    metric_rows, comparison_rows = [], []
    rng = np.random.default_rng(SEED_BOOTSTRAP)
    for label, mask in [("10", snr == 10), ("15", snr == 15), ("20", snr == 20), ("All", np.ones(len(snr), bool))]:
        for column, parameter in enumerate(parameters):
            dnn = pred[f"{parameter.lower()}_dnn"].astype(np.float64)
            hdelm = pred[f"{parameter.lower()}_hdelm"].astype(np.float64)
            for model, estimate in (("DNN-G", dnn), ("H-DELM V2", hdelm)):
                metric_rows.append({
                    "SNR": label,
                    "N": int(mask.sum()),
                    "Model": model,
                    "Parameter": parameter,
                    **metrics(truth[mask, column], estimate[mask]),
                })
            dnn_metrics = metrics(truth[mask, column], dnn[mask])
            hdelm_metrics = metrics(truth[mask, column], hdelm[mask])
            comparison_rows.append({
                "SNR": label,
                "Parameter": parameter,
                "DNN-G RMSE": dnn_metrics["RMSE"],
                "H-DELM RMSE": hdelm_metrics["RMSE"],
                "Relative RMSE improvement (%)": (
                    (dnn_metrics["RMSE"] - hdelm_metrics["RMSE"]) / dnn_metrics["RMSE"] * 100
                ),
            })

    metrics_df = pd.DataFrame(metric_rows)
    comp_df = pd.DataFrame(comparison_rows)
    metrics_df.to_csv(RESULTS / "final_metrics.csv", index=False)
    comp_df.to_csv(RESULTS / "per_snr_comparison.csv", index=False)
    for parameter, filename in (("CBF", "cbf_performance.csv"), ("ATT", "att_performance.csv")):
        metrics_df[(metrics_df["Parameter"] == parameter) & (metrics_df["SNR"] != "All")].to_csv(
            TABLES / filename, index=False
        )
    metrics_df[metrics_df["SNR"] == "All"].to_csv(TABLES / "overall_performance.csv", index=False)

    architecture_rows = [
        ["DNN input", "4 raw ASL signal samples at the configured PLDs", "Shared input to each target model"],
        ["DNN hidden stack", "4 fully connected layers × 64 units", "Separate frozen CBF and ATT networks"],
        ["DNN activation", "ELU", "Every hidden layer"],
        ["DNN initialization", "Kaiming normal weights; zero biases", "As implemented in DNN_G_Net"],
        ["DNN output", "Linear logit → sigmoid → physical range", "CBF [0,100]; ATT [0.5,3.0]"],
        ["DNN learned representation", "64-dimensional penultimate activation", "Used by target-specific ELM"],
        ["Additional ELM features", "15 signal-derived features", "Four samples, slopes, ratios, moments, extrema, AUC"],
        ["ELM input dimension", "79", "64 DNN + 15 signal features"],
        ["ELM hidden layer", "512 fixed random neurons; tanh", "Uniform W/b initialization; seed 42"],
        ["ELM output", "Ridge regression on residual", "Saved artifacts use λ=0.01"],
        ["Residual target", "y_true − y_DNN", "Fit on independent calibration set"],
        ["Final output", "clip(y_DNN + predicted residual)", "Clipped to physiological target range"],
        ["Reference match", "Not verifiable", "Reference H-DELM paper unavailable in workspace"],
    ]
    pd.DataFrame(architecture_rows, columns=["Component", "Verified implementation", "Notes"]).to_csv(
        TABLES / "architecture_table.csv", index=False
    )

    configuration_rows = [
        ["PLDs (s)", "[1.525, 2.025, 2.525, 3.025]", "Source simulator"],
        ["CBF range", "[0, 100] ml/100g/min", "Uniform sampling"],
        ["ATT range", "[0.5, 3.0] s", "Uniform sampling"],
        ["Signal model", "Buxton general kinetic model", "Repository implementation"],
        ["Noise", "Dual-channel quadrature Rician magnitude", "sigma = SIG_REF_SC / SNR"],
        ["DNN training observations", "1,000,000", "10,000 pairs × 100 SNR levels; generated in-memory"],
        ["DNN validation observations", "100,000", "1,000 pairs × 100 SNR levels"],
        ["ELM calibration observations", "100,000", "80,000 fit + 20,000 validation"],
        ["Independent test observations", "30,000", "10,000 at each SNR 10, 15, 20"],
        ["Training seed", "42", "V1 DNN-G generator"],
        ["Validation seed", "99", "V1 DNN-G generator"],
        ["Calibration seed", "31415", "Stored in calibration dataset"],
        ["Test seed", str(int(pred["seed"])), "Fresh final test cohort"],
        ["ELM seed", "42", "Fixed random hidden weights/biases"],
        ["Input normalization", "Mean/std fitted on DNN training cohort", "Same values applied at inference"],
        ["Provenance qualification", "Training cohort generated in-memory", "Exact train arrays/digest not persisted"],
        ["Ridge sweep qualification", "Sweep favored λ=1.0; final artifact λ=0.01", "Final saved model evaluated as-is"],
    ]
    pd.DataFrame(configuration_rows, columns=["Setting", "Value", "Evidence/qualification"]).to_csv(
        TABLES / "experimental_configuration.csv", index=False
    )

    stat_rows = []
    for level in levels:
        mask = snr == level
        for parameter, column in (("CBF", 0), ("ATT", 1)):
            dnn = pred[f"{parameter.lower()}_dnn"].astype(np.float64)[mask]
            hdelm = pred[f"{parameter.lower()}_hdelm"].astype(np.float64)[mask]
            true = truth[mask, column]
            dnn_squared_error = (dnn - true) ** 2
            hdelm_squared_error = (hdelm - true) ** 2
            test = wilcoxon(dnn_squared_error, hdelm_squared_error, alternative="two-sided", method="approx")
            low, high = bootstrap_rmse_improvement(true, dnn, hdelm, rng)
            stat_rows.append({
                "SNR": str(int(level)),
                "Parameter": parameter,
                "Paired RMSE improvement (DNN−H-DELM)": metrics(true, dnn)["RMSE"] - metrics(true, hdelm)["RMSE"],
                "Bootstrap 95% CI lower": low,
                "Bootstrap 95% CI upper": high,
                "Bootstrap replicates": N_BOOTSTRAP,
                "Bootstrap seed": SEED_BOOTSTRAP,
                "Test": "Paired Wilcoxon signed-rank on per-observation squared errors",
                "Raw p-value": float(test.pvalue),
            })
    adjusted = holm_adjust([row["Raw p-value"] for row in stat_rows])
    for row, value in zip(stat_rows, adjusted):
        row["Holm-adjusted p-value"] = float(value)
        row["Significant at 0.05"] = bool(value < 0.05)
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(TABLES / "statistical_comparison.csv", index=False)
    stat_df.to_csv(RESULTS / "paired_statistical_tests.csv", index=False)

    train_stats = [
        ["DNN training", "CBF", 1_000_000, 0.0314419642, 99.9977035522, 49.7121505737, 28.8206501007, "ml/100g/min"],
        ["DNN training", "ATT", 1_000_000, 0.5000329018, 2.9998681545, 1.7573384047, 0.7223210931, "s"],
        ["DNN training", "SNR", 1_000_000, 10.0, 20.0, np.nan, np.nan, "100 equally spaced levels; 10k each"],
        ["DNN validation", "CBF", 100_000, 0.0336519405, 99.9687271118, 52.2152290344, 28.3976573944, "ml/100g/min"],
        ["DNN validation", "ATT", 100_000, 0.5016798973, 2.9951076508, 1.7646943331, 0.7338863015, "s"],
        ["DNN validation", "SNR", 100_000, 10.0, 20.0, np.nan, np.nan, "100 equally spaced levels; 1k each"],
    ]
    pd.DataFrame(train_stats, columns=["Split", "Variable", "N", "Minimum", "Maximum", "Mean", "SD", "Units/distribution"]).to_csv(
        DATASETS / "dataset_statistics.csv", index=False
    )

    dataset_meta = {
        "source": "Existing V1 DNN-G pipeline generator, independently rerun for summary statistics",
        "training_observations": 1_000_000,
        "training_inputs_shape": [1_000_000, 4],
        "training_targets_shape": [1_000_000, 2],
        "validation_observations": 100_000,
        "calibration_observations": 100_000,
        "test_observations": 30_000,
        "training_arrays_persisted": False,
        "training_provenance_note": "V1 generator constructs training arrays in memory; no dataset digest was saved with the model checkpoints.",
        "training_seed": 42,
        "validation_seed": 99,
        "calibration_seed": 31415,
        "test_seed": int(pred["seed"]),
        "plds_seconds": [1.525, 2.025, 2.525, 3.025],
        "snr_test_counts": {"10": 10000, "15": 10000, "20": 10000},
    }
    (DATASETS / "dataset_metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
    return metrics_df, comp_df, stat_df


def save_figure(fig, stem):
    for kind, directory in FIG_DIRS.items():
        fig.savefig(directory / f"{stem}.{kind}", dpi=600 if kind == "png" else None)
    plt.close(fig)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=8.5):
        patch = matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#38434f", linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2, color="#38434f"):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.25, "color": color})

    box(0.25, 2.9, 1.7, 1.1, "Simulated ASL signals\n4 PLDs: 1.525–3.025 s", "#e7edf4")
    box(2.5, 4.65, 2.2, 1.1, "Frozen DNN-G\n4 × 64 ELU layers", "#dceaf7")
    box(5.35, 4.65, 2.0, 1.1, "DNN prediction\nCBF and ATT heads", "#dceaf7")
    box(2.5, 1.05, 2.2, 1.1, "Signal features\n15 physics-derived values", "#e4f2e7")
    box(5.35, 2.9, 2.0, 1.2, "Concatenate\n64 DNN + 15 signal\n= 79 features", "#f0e9f8")
    box(8.0, 2.9, 2.0, 1.2, "Fixed ELM hidden layer\n512 random tanh units", "#fff0d9")
    box(10.55, 2.9, 2.0, 1.2, "Ridge readout\nλ = 0.01 saved model", "#fff0d9")
    box(8.0, 5.2, 2.0, 0.9, "Residual target during calibration\ny_true − y_DNN", "#fbe3e3")
    box(10.55, 0.65, 2.0, 0.9, "Add correction and clip\ny_final = y_DNN + r̂", "#e4f2e7")
    box(10.55, 1.9, 2.0, 0.7, "Predicted residual r̂", "#fbe3e3")

    arrow(1.95, 3.65, 2.5, 5.15)
    arrow(1.95, 3.2, 2.5, 1.6)
    arrow(4.7, 5.15, 5.35, 5.15)
    arrow(4.7, 5.0, 5.35, 3.85)
    arrow(4.7, 1.6, 5.35, 3.15)
    arrow(7.35, 3.5, 8.0, 3.5)
    arrow(10.0, 3.5, 10.55, 3.5)
    arrow(10.0, 5.65, 10.55, 4.1)
    arrow(11.55, 2.9, 11.55, 2.6)
    arrow(11.55, 1.9, 11.55, 1.55)
    ax.plot([7.35, 7.7, 12.8, 12.8], [5.2, 6.4, 6.4, 1.1], color="#286090", lw=1.25)
    arrow(12.8, 1.1, 12.55, 1.1, color="#286090")
    ax.text(0.35, 6.35, "Blue path: DNN estimate passed directly to final output", color="#286090", fontsize=8.5)
    ax.set_title("Implemented H-DELM V2 inference and residual-calibration flow", pad=10)
    fig.tight_layout()
    save_figure(fig, "fig01_hdelm_architecture")


def generate_figures():
    pred = load_predictions()
    truth = pred["Y_test"].astype(np.float64)
    snr = pred["snr_test"]
    levels = (10.0, 15.0, 20.0)
    parameters = (("CBF", 0, "ml/100g/min"), ("ATT", 1, "s"))
    metrics_df, comp_df, stat_df = prepare_tables()

    architecture_figure()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, prefix, title in zip(axes, ("cbf", "att"), ("CBF DNN-G", "ATT DNN-G")):
        hist = pd.read_csv(ROOT / "models" / "model_metadata" / f"{prefix}_dnn_g_training_history.csv")
        ax.plot(hist["epoch"], hist["train_loss"], label="Training loss", color="#276FBF", lw=1.5)
        ax.plot(hist["epoch"], hist["val_loss"], label="Validation loss", color="#D1495B", lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE loss")
        ax.set_title(f"{title} ({len(hist)} epochs)")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("DNN-G training and validation convergence")
    fig.tight_layout()
    save_figure(fig, "fig02_training_validation_loss")

    for parameter, index, units in parameters:
        rows = comp_df[(comp_df["Parameter"] == parameter) & (comp_df["SNR"] != "All")]
        fig, ax = plt.subplots(figsize=(5.4, 3.7))
        x = np.arange(3)
        width = 0.34
        ax.bar(x - width / 2, rows["DNN-G RMSE"], width, label="DNN-G", color="#276FBF")
        ax.bar(x + width / 2, rows["H-DELM RMSE"], width, label="H-DELM V2", color="#D1495B")
        ax.set_xticks(x, [str(v) for v in levels])
        ax.set_xlabel("SNR")
        ax.set_ylabel(f"{parameter} RMSE ({units})")
        ax.set_title(f"{parameter} RMSE by SNR (n=10,000 per SNR)")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        save_figure(fig, f"fig0{3 if parameter == 'CBF' else 4}_{parameter.lower()}_rmse_vs_snr")

    rng = np.random.default_rng(20261103)
    for parameter, index, units in parameters:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True, sharey=True)
        for ax, model, key, color in zip(
            axes, ("DNN-G", "H-DELM V2"), (f"{parameter.lower()}_dnn", f"{parameter.lower()}_hdelm"), ("#276FBF", "#D1495B")
        ):
            sample = rng.choice(len(truth), min(8000, len(truth)), replace=False)
            y = truth[sample, index]
            estimate = pred[key][sample]
            ax.scatter(y, estimate, s=5, alpha=0.16, color=color, rasterized=True, edgecolors="none")
            lo, hi = float(np.min(truth[:, index])), float(np.max(truth[:, index]))
            ax.plot([lo, hi], [lo, hi], "--", color="#444444", lw=1, label="Identity")
            slope, intercept = np.polyfit(y, estimate, 1)
            xx = np.linspace(lo, hi, 100)
            ax.plot(xx, slope * xx + intercept, color="#3A7D44", lw=1.1, label="Least-squares line")
            m = metrics(truth[:, index], pred[key])
            ax.text(0.04, 0.96, f"RMSE={m['RMSE']:.4f} {units}\nMAE={m['MAE']:.4f} {units}\nCCC={m['Lin_CCC']:.4f}\nn={len(truth):,}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"})
            ax.set_title(model)
            ax.set_xlabel(f"Ground-truth {parameter} ({units})")
            ax.grid(alpha=0.15)
            ax.legend(frameon=False, loc="lower right")
        axes[0].set_ylabel(f"Predicted {parameter} ({units})")
        fig.suptitle(f"{parameter} estimation versus ground truth — independent simulated test")
        fig.tight_layout()
        save_figure(fig, f"fig0{5 if parameter == 'CBF' else 6}_{parameter.lower()}_prediction_scatter")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (parameter, index, units), model, key in zip(
        axes.flat,
        (parameters[0], parameters[0], parameters[1], parameters[1]),
        ("DNN-G", "H-DELM V2", "DNN-G", "H-DELM V2"),
        ("cbf_dnn", "cbf_hdelm", "att_dnn", "att_hdelm"),
    ):
        estimate = pred[key].astype(np.float64)
        average = (truth[:, index] + estimate) / 2
        difference = estimate - truth[:, index]
        bias = float(np.mean(difference))
        loa = 1.96 * float(np.std(difference, ddof=1))
        ax.scatter(average, difference, s=4, alpha=0.12, rasterized=True, color="#276FBF" if model == "DNN-G" else "#D1495B")
        ax.axhline(bias, color="#222222", lw=1.2, label=f"Bias {bias:.4f}")
        ax.axhline(bias + loa, color="#777777", linestyle="--", lw=1, label=f"+1.96 SD {bias+loa:.4f}")
        ax.axhline(bias - loa, color="#777777", linestyle="--", lw=1, label=f"−1.96 SD {bias-loa:.4f}")
        ax.set_title(f"{parameter} — {model}")
        ax.set_xlabel(f"Mean of estimate and truth ({units})")
        ax.set_ylabel(f"Estimate − truth ({units})")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.15)
    fig.suptitle("Bland–Altman simulated-data agreement analysis (not clinical measurements)")
    fig.tight_layout()
    save_figure(fig, "fig07_bland_altman_simulated")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, (parameter, index, units) in enumerate(parameters):
        keys = (f"{parameter.lower()}_dnn", f"{parameter.lower()}_hdelm")
        for ax, level in zip(axes[row], levels):
            mask = snr == level
            for key, model, color in zip(keys, ("DNN-G", "H-DELM V2"), ("#276FBF", "#D1495B")):
                errors = pred[key][mask] - truth[mask, index]
                ax.hist(errors, bins=60, density=True, histtype="step", lw=1.3, label=model, color=color)
            ax.axvline(0, color="#333333", linestyle="--", lw=0.9)
            ax.set_title(f"{parameter}, SNR {int(level)}")
            ax.set_xlabel(f"Prediction error ({units})")
            ax.set_ylabel("Density")
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Prediction error distributions by SNR")
    fig.tight_layout()
    save_figure(fig, "fig08_error_distributions")

    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    for col, (parameter, index, units) in enumerate(parameters):
        dnn = pred[f"{parameter.lower()}_dnn"].astype(np.float64)
        hdelm = pred[f"{parameter.lower()}_hdelm"].astype(np.float64)
        actual_resid = truth[:, index] - dnn
        predicted_resid = pred[f"{parameter.lower()}_residual"].astype(np.float64)
        err_before = dnn - truth[:, index]
        err_after = hdelm - truth[:, index]
        sample = rng.choice(len(truth), 8000, replace=False)
        ax = axes[0, col]
        ax.scatter(truth[sample, index], actual_resid[sample], s=4, alpha=0.14, rasterized=True)
        ax.axhline(0, color="#333333", linestyle="--", lw=0.8)
        ax.set_title(f"{parameter}: DNN residual vs truth")
        ax.set_xlabel(f"Ground truth ({units})")
        ax.set_ylabel(f"True − DNN ({units})")

        ax = axes[1, col]
        ax.scatter(actual_resid[sample], predicted_resid[sample], s=4, alpha=0.14, rasterized=True, color="#D1495B")
        limits = [min(actual_resid.min(), predicted_resid.min()), max(actual_resid.max(), predicted_resid.max())]
        ax.plot(limits, limits, "--", color="#333333", lw=0.8)
        corr = float(np.corrcoef(actual_resid, predicted_resid)[0, 1])
        ax.set_title(f"ELM residual vs actual (r={corr:.3f})")
        ax.set_xlabel(f"Actual residual ({units})")
        ax.set_ylabel(f"Predicted residual ({units})")

        ax = axes[2, col]
        ax.scatter(err_before[sample], err_after[sample], s=4, alpha=0.14, rasterized=True, color="#3A7D44")
        limits = [min(err_before.min(), err_after.min()), max(err_before.max(), err_after.max())]
        ax.plot(limits, limits, "--", color="#333333", lw=0.8)
        ax.set_title(f"Error before vs after correction")
        ax.set_xlabel(f"DNN error ({units})")
        ax.set_ylabel(f"H-DELM error ({units})")

        ax = axes[3, col]
        ax.hist(err_before, bins=70, density=True, histtype="step", lw=1.3, label="DNN-G", color="#276FBF")
        ax.hist(err_after, bins=70, density=True, histtype="step", lw=1.3, label="H-DELM V2", color="#D1495B")
        ax.axvline(0, color="#333333", linestyle="--", lw=0.8)
        ax.set_title(f"Overall error distributions")
        ax.set_xlabel(f"Prediction − truth ({units})")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    fig.suptitle("Residual correction diagnostics — same 30,000 simulated observations")
    fig.tight_layout()
    save_figure(fig, "fig09_elm_residual_correction")

    rows = comp_df[(comp_df["SNR"] != "All")]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xlabels = [f"{p}\nSNR {int(s)}" for s in levels for p in ("CBF", "ATT")]
    improvements = []
    for level in levels:
        for parameter, _, _ in parameters:
            improvements.append(float(rows[(rows["SNR"] == str(int(level))) & (rows["Parameter"] == parameter)]["Relative RMSE improvement (%)"].iloc[0]))
    x = np.arange(6)
    colors = ["#3A7D44" if value >= 0 else "#D1495B" for value in improvements]
    ax.bar(x, improvements, color=colors, width=0.68)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x, xlabels)
    ax.set_ylabel("Relative RMSE improvement (%)")
    ax.set_title("H-DELM V2 relative RMSE change versus DNN-G")
    ax.grid(axis="y", alpha=0.18)
    for xi, value in zip(x, improvements):
        ax.text(xi, value + (0.035 if value >= 0 else -0.035), f"{value:+.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig10_relative_rmse_improvement")

    fig, ax = plt.subplots(figsize=(8, 4.7))
    y_positions = np.arange(len(stat_df))[::-1]
    values = stat_df["Paired RMSE improvement (DNN−H-DELM)"].to_numpy()
    low = stat_df["Bootstrap 95% CI lower"].to_numpy()
    high = stat_df["Bootstrap 95% CI upper"].to_numpy()
    ax.errorbar(values, y_positions, xerr=[values - low, high - values], fmt="o", capsize=3, color="#276FBF")
    for i, row in stat_df.reset_index(drop=True).iterrows():
        label = f"{row['Parameter']} SNR {int(row['SNR'])}: Holm p={row['Holm-adjusted p-value']:.2g}"
        if row["Significant at 0.05"]:
            label += " *"
        ax.text(high[i] + 0.015, y_positions[i], label, va="center", fontsize=7.5)
    ax.axvline(0, color="#333333", linestyle="--", lw=0.9)
    ax.set_yticks(y_positions, [f"{r.Parameter}, SNR {int(r.SNR)}" for r in stat_df.itertuples()])
    ax.set_xlabel("Paired RMSE improvement: DNN-G − H-DELM (positive favors H-DELM)")
    ax.set_title(f"Paired bootstrap 95% intervals ({N_BOOTSTRAP:,} resamples); Wilcoxon p-values Holm-adjusted")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    save_figure(fig, "fig11_statistical_significance")
    return 11
