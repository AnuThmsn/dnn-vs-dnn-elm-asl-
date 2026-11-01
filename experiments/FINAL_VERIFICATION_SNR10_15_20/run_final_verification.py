"""Generate and evaluate a fresh, independent SNR 10/15/20 test cohort."""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V1 = ROOT / "experiments" / "paper_reproduction_snr10_15_20"
V2 = ROOT / "experiments" / "paper_reproduction_snr10_15_20_v2"
sys.path.insert(0, str(ROOT))

from experiments.physics import (  # noqa: E402
    PLDS,
    TAU,
    CBF_MIN,
    CBF_MAX,
    ATT_MIN,
    ATT_MAX,
    compute_physical_signals,
    add_rician_noise,
)
from experiments.paper_reproduction_snr10_15_20_v2.standalone_inference import (  # noqa: E402
    StandaloneASLEstimator,
)

SEED = 20261101
SNR_LEVELS = (10.0, 15.0, 20.0)
N_PER_SNR = 10_000


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    denom = var_true + var_pred + (np.mean(y_true) - np.mean(y_pred)) ** 2
    ccc = 2.0 * np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))) / denom
    mse = float(np.mean(err**2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(err))),
        "Bias": float(np.mean(err)),
        "Lin_CCC": float(ccc),
    }


def training_label_pairs(seed, unique_pairs, repeats):
    rng = np.random.default_rng(seed)
    cbf = rng.uniform(CBF_MIN, CBF_MAX, unique_pairs).astype(np.float64)
    att = rng.uniform(ATT_MIN, ATT_MAX, unique_pairs).astype(np.float64)
    return np.column_stack(
        [np.repeat(cbf, repeats), np.repeat(att, repeats)]
    ).astype(np.float32)


def tuple_set(values):
    return set(map(tuple, np.asarray(values, dtype=np.float32)))


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def plot_results(pred, truth, snr, comparison):
    snrs = list(SNR_LEVELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, parameter in zip(axes, ("CBF", "ATT")):
        rows = comparison[
            (comparison["Parameter"] == parameter) & (comparison["SNR"] != "All")
        ]
        x = np.arange(len(snrs))
        width = 0.36
        ax.bar(x - width / 2, rows["DNN-G RMSE"], width, label="DNN-G")
        ax.bar(x + width / 2, rows["H-DELM RMSE"], width, label="H-DELM V2")
        ax.set_xticks(x, [str(s) for s in snrs])
        ax.set_xlabel("SNR")
        ax.set_ylabel(f"{parameter} RMSE")
        ax.set_title(f"{parameter} RMSE by SNR")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "PER_SNR_RMSE_COMPARISON.png", dpi=180)
    plt.close(fig)

    rng = np.random.default_rng(SEED + 1)
    for column, parameter in enumerate(("CBF", "ATT")):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
        for row, model in enumerate(("DNN-G", "H-DELM")):
            for col, level in enumerate(snrs):
                ax = axes[row, col]
                ids = np.flatnonzero(snr == level)
                sample_ids = rng.choice(ids, size=min(2500, len(ids)), replace=False)
                model_key = "dnn" if model == "DNN-G" else "hdelm"
                predicted = pred[f"{parameter.lower()}_{model_key}"]
                ax.scatter(truth[sample_ids, column], predicted[sample_ids], s=3, alpha=0.22)
                lower = float(np.min(truth[ids, column]))
                upper = float(np.max(truth[ids, column]))
                ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
                ax.set_title(f"{model}, SNR {int(level)}")
                if col == 0:
                    ax.set_ylabel(f"Predicted {parameter}")
                if row == 1:
                    ax.set_xlabel(f"True {parameter}")
                ax.grid(alpha=0.2)
        fig.suptitle(f"{parameter}: predictions versus ground truth")
        fig.tight_layout()
        fig.savefig(OUT / f"{parameter}_PREDICTION_VS_GROUND_TRUTH.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, level in zip(axes, snrs):
            ids = snr == level
            for model, color in (("DNN-G", "tab:blue"), ("H-DELM", "tab:orange")):
                model_key = "dnn" if model == "DNN-G" else "hdelm"
                errors = pred[f"{parameter.lower()}_{model_key}"][ids] - truth[ids, column]
                ax.hist(errors, bins=70, density=True, histtype="step", linewidth=1.5, label=model, color=color)
            ax.set_title(f"SNR {int(level)}")
            ax.set_xlabel(f"{parameter} prediction error")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle(f"{parameter}: residual distributions")
        fig.tight_layout()
        fig.savefig(OUT / f"{parameter}_RESIDUAL_DISTRIBUTIONS.png", dpi=180)
        plt.close(fig)


def main():
    log_path = OUT / "EXECUTION_LOG.txt"
    log_path.write_text(
        f"Final independent verification started: {datetime.now(timezone.utc).isoformat()}\n"
        f"Command: python {Path(__file__).relative_to(ROOT)}\n"
        f"Test seed: {SEED}; samples per SNR: {N_PER_SNR}\n",
        encoding="utf-8",
    )

    required = [
        V1 / "models" / "cbf_dnn_g_best.pth",
        V1 / "models" / "att_dnn_g_best.pth",
        V1 / "results" / "input_normalization_stats.npz",
        V2 / "models" / "cbf_hdelm_v2_artifact.npz",
        V2 / "models" / "att_hdelm_v2_artifact.npz",
        V2 / "results" / "data" / "residual_calibration_100k.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen artifact(s) missing: " + ", ".join(missing))

    rng = np.random.default_rng(SEED)
    X_chunks, Y_chunks, SNR_chunks = [], [], []
    for level in SNR_LEVELS:
        cbf = rng.uniform(CBF_MIN, CBF_MAX, N_PER_SNR).astype(np.float64)
        att = rng.uniform(ATT_MIN, ATT_MAX, N_PER_SNR).astype(np.float64)
        clean = compute_physical_signals(PLDS, TAU, cbf, att)
        noisy = add_rician_noise(clean, level, rng)
        X_chunks.append(noisy)
        Y_chunks.append(np.column_stack([cbf, att]).astype(np.float32))
        SNR_chunks.append(np.full(N_PER_SNR, level, dtype=np.float64))

    X_test = np.concatenate(X_chunks).astype(np.float32)
    Y_test = np.concatenate(Y_chunks).astype(np.float32)
    snr_test = np.concatenate(SNR_chunks)
    assert X_test.shape == (30_000, 4)
    assert Y_test.shape == (30_000, 2)
    assert np.isfinite(X_test).all() and np.isfinite(Y_test).all()
    assert np.all((Y_test[:, 0] >= CBF_MIN) & (Y_test[:, 0] <= CBF_MAX))
    assert np.all((Y_test[:, 1] >= ATT_MIN) & (Y_test[:, 1] <= ATT_MAX))
    counts = {str(int(level)): int(np.count_nonzero(snr_test == level)) for level in SNR_LEVELS}
    assert counts == {"10": 10_000, "15": 10_000, "20": 10_000}

    cal = np.load(V2 / "results" / "data" / "residual_calibration_100k.npz")
    cal_labels = cal["Y_cal"]
    train_labels = training_label_pairs(seed=42, unique_pairs=10_000, repeats=100)
    val_labels = training_label_pairs(seed=99, unique_pairs=1_000, repeats=100)
    test_pairs = tuple_set(Y_test)
    overlap = {
        "train_exact_parameter_pair_intersections": len(test_pairs & tuple_set(train_labels)),
        "validation_exact_parameter_pair_intersections": len(test_pairs & tuple_set(val_labels)),
        "calibration_exact_parameter_pair_intersections": len(test_pairs & tuple_set(cal_labels)),
    }
    assert all(value == 0 for value in overlap.values()), f"Test label overlap found: {overlap}"
    assert SEED not in (42, 99, 31415, 2026)

    estimator = StandaloneASLEstimator(
        cbf_dnn_path=str(V1 / "models" / "cbf_dnn_g_best.pth"),
        att_dnn_path=str(V1 / "models" / "att_dnn_g_best.pth"),
        cbf_elm_path=str(V2 / "models" / "cbf_hdelm_v2_artifact.npz"),
        att_elm_path=str(V2 / "models" / "att_hdelm_v2_artifact.npz"),
        norm_stats_path=str(V1 / "results" / "input_normalization_stats.npz"),
    )
    output = estimator.predict(X_test)
    predictions = {
        "cbf_dnn": output["cbf_dnn"],
        "att_dnn": output["att_dnn"],
        "cbf_hdelm": output["cbf_hdelm"],
        "att_hdelm": output["att_hdelm"],
    }
    assert all(np.isfinite(values).all() for values in predictions.values())
    assert np.all((predictions["cbf_dnn"] >= CBF_MIN) & (predictions["cbf_dnn"] <= CBF_MAX))
    assert np.all((predictions["cbf_hdelm"] >= CBF_MIN) & (predictions["cbf_hdelm"] <= CBF_MAX))
    assert np.all((predictions["att_dnn"] >= ATT_MIN) & (predictions["att_dnn"] <= ATT_MAX))
    assert np.all((predictions["att_hdelm"] >= ATT_MIN) & (predictions["att_hdelm"] <= ATT_MAX))

    metric_rows, comparison_rows, statistical_rows = [], [], []
    for label, mask in [("10", snr_test == 10), ("15", snr_test == 15), ("20", snr_test == 20), ("All", np.ones(len(snr_test), dtype=bool))]:
        for index, parameter in enumerate(("CBF", "ATT")):
            dnn = predictions[f"{parameter.lower()}_dnn"]
            hdelm = predictions[f"{parameter.lower()}_hdelm"]
            for model, estimate in (("DNN-G", dnn), ("H-DELM", hdelm)):
                metric_rows.append(
                    {"SNR": label, "N": int(mask.sum()), "Model": model, "Parameter": parameter,
                     **metrics(Y_test[mask, index], estimate[mask])}
                )
            dnn_rmse = metrics(Y_test[mask, index], dnn[mask])["RMSE"]
            hdelm_rmse = metrics(Y_test[mask, index], hdelm[mask])["RMSE"]
            comparison_rows.append({
                "SNR": label,
                "Parameter": parameter,
                "DNN-G RMSE": dnn_rmse,
                "H-DELM RMSE": hdelm_rmse,
                "Relative RMSE improvement (%)": (dnn_rmse - hdelm_rmse) / dnn_rmse * 100.0,
            })
            if label != "All":
                err_dnn = (dnn[mask] - Y_test[mask, index]).astype(np.float64) ** 2
                err_hdelm = (hdelm[mask] - Y_test[mask, index]).astype(np.float64) ** 2
                stat = wilcoxon(err_dnn, err_hdelm, alternative="two-sided", method="approx")
                statistical_rows.append({
                    "SNR": label,
                    "Parameter": parameter,
                    "Test": "Paired Wilcoxon signed-rank on per-observation squared errors",
                    "p_value": float(stat.pvalue),
                    "median_squared_error_difference_DNN_minus_HDELM": float(np.median(err_dnn - err_hdelm)),
                })

    adjusted = holm_adjust([row["p_value"] for row in statistical_rows])
    for row, p_adj in zip(statistical_rows, adjusted):
        row["holm_adjusted_p_value"] = float(p_adj)
        row["significant_at_0.05"] = bool(p_adj < 0.05)

    metric_df = pd.DataFrame(metric_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    metric_df.to_csv(OUT / "FINAL_METRICS.csv", index=False)
    comparison_df.to_csv(OUT / "FINAL_PER_SNR_COMPARISON.csv", index=False)
    pd.DataFrame(statistical_rows).to_csv(OUT / "PAIRED_STATISTICAL_TESTS.csv", index=False)
    np.savez_compressed(
        OUT / "FINAL_PREDICTIONS.npz",
        X_test=X_test,
        Y_test=Y_test,
        snr_test=snr_test,
        cbf_dnn=predictions["cbf_dnn"],
        att_dnn=predictions["att_dnn"],
        cbf_hdelm=predictions["cbf_hdelm"],
        att_hdelm=predictions["att_hdelm"],
        cbf_residual=output["cbf_residual"],
        att_residual=output["att_residual"],
        seed=np.asarray(SEED),
        plds=PLDS,
    )
    plot_results(predictions, Y_test, snr_test, comparison_df)

    artifact_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in required}
    output_hashes = {
        name: sha256(OUT / name)
        for name in (
            "FINAL_PREDICTIONS.npz",
            "FINAL_METRICS.csv",
            "FINAL_PER_SNR_COMPARISON.csv",
            "PAIRED_STATISTICAL_TESTS.csv",
            "PER_SNR_RMSE_COMPARISON.png",
            "CBF_PREDICTION_VS_GROUND_TRUTH.png",
            "ATT_PREDICTION_VS_GROUND_TRUTH.png",
            "CBF_RESIDUAL_DISTRIBUTIONS.png",
            "ATT_RESIDUAL_DISTRIBUTIONS.png",
        )
    }
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "test_seed": SEED,
        "training_seed_documented_in_pipeline": 42,
        "validation_seed_documented_in_pipeline": 99,
        "calibration_seed_stored_in_dataset": int(cal["seed"]),
        "calibration_shape": list(cal["X_cal"].shape),
        "calibration_snr_range": [float(np.min(cal["snr_cal"])), float(np.max(cal["snr_cal"]))],
        "legacy_test_seed": 2026,
        "snr_levels": list(SNR_LEVELS),
        "samples_per_snr": counts,
        "total_samples": int(len(X_test)),
        "input_shape": list(X_test.shape),
        "target_shape": list(Y_test.shape),
        "plds_seconds": PLDS.tolist(),
        "cbf_range": [CBF_MIN, CBF_MAX],
        "att_range_seconds": [ATT_MIN, ATT_MAX],
        "simulator": "experiments.physics.compute_physical_signals + add_rician_noise",
        "noise_model": "dual-channel quadrature Rician magnitude; sigma=SIG_REF_SC/SNR",
        "finite_signals_and_labels": True,
        "labels_in_physiological_bounds": True,
        "exact_test_label_pair_overlap": overlap,
        "overlap_check_scope": "Exact CBF/ATT label-pair intersections against source-pipeline regenerated train/validation parameter pairs and stored calibration labels; not a cryptographic proof of historical non-use.",
        "models_loaded": [
            "Frozen DNN-G CBF/ATT checkpoints from V1",
            "Saved H-DELM V2 ELM artifacts (W, b, beta, feature scaler)",
        ],
        "no_test_fit_or_tuning": True,
        "artifact_sha256": artifact_hashes,
        "output_sha256": output_hashes,
        "runtime": {"numpy": np.__version__, "torch": torch.__version__, "scipy": scipy.__version__, "cuda_available": torch.cuda.is_available()},
        "outputs": {
            "predictions": "FINAL_PREDICTIONS.npz",
            "metrics": "FINAL_METRICS.csv",
            "comparison": "FINAL_PER_SNR_COMPARISON.csv",
            "paired_tests": "PAIRED_STATISTICAL_TESTS.csv",
        },
    }
    (OUT / "FINAL_TEST_DATA_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(
            f"Completed: {datetime.now(timezone.utc).isoformat()}\n"
            "Status: SUCCESS; generated fresh 30k cohort, verified labels and signal bounds, "
            "loaded frozen DNN/ELM artifacts, ran paired inference, saved CSV/NPZ/plots and Holm-adjusted tests.\n"
        )
    print(comparison_df.to_string(index=False))
    print(pd.DataFrame(statistical_rows).to_string(index=False))
    print(f"Saved final independent evaluation artifacts to {OUT}")


if __name__ == "__main__":
    main()
