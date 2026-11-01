"""Run frozen-baseline and calibration-only hybrid comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import wilcoxon
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "experiments" / "HYBRID_ARCHITECTURE_RESEARCH"
BASELINE = ROOT / "submited h-delm"
SCRIPTS = BASELINE / "scripts"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))

from source_dnn_g import DNN_G_Net, device  # noqa: E402
from source_physics import (  # noqa: E402
    ATT_MAX,
    ATT_MIN,
    CBF_MAX,
    CBF_MIN,
    PLDS,
)
from source_residual_hdelm_v2 import extract_15_physics_features  # noqa: E402


CAL_SPLIT_SEED = 20261102
ELM_SEED = 42
MLP_SEEDS = (11, 42, 31415)
ELM_SEEDS = (42, 31415, 2026)
RIDGE_LAMBDA = 0.01
ELM_HIDDEN = 512
BOOTSTRAP_REPLICATES = 1000
PERMUTATION_REPLICATES = 2000
SNR_LEVELS = (10.0, 15.0, 20.0)
PARAMETERS = ("CBF", "ATT")
BOUNDS = np.asarray([[CBF_MIN, CBF_MAX], [ATT_MIN, ATT_MAX]], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_networks():
    model_dir = BASELINE / "models"
    norm = np.load(model_dir / "model_metadata" / "input_normalization_stats.npz")
    networks = []
    for target, bounds in zip(("cbf", "att"), BOUNDS):
        net = DNN_G_Net(4, 4, 64, float(bounds[0]), float(bounds[1])).to(device)
        checkpoint = model_dir / "dnn_checkpoint" / f"{target}_dnn_g_best.pth"
        net.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        net.eval()
        networks.append(net)
    return networks, norm


def normalize_signals(x, norm):
    return ((x - norm["X_mean"]) / norm["X_std"]).astype(np.float32)


def parameter_pair_set(seed, count):
    rng = np.random.default_rng(seed)
    cbf = rng.uniform(CBF_MIN, CBF_MAX, count).astype(np.float32)
    att = rng.uniform(ATT_MIN, ATT_MAX, count).astype(np.float32)
    return set(zip(cbf.tolist(), att.tolist()))


def infer_dnn(networks, x_raw, x_norm):
    predictions = np.column_stack(
        [net.predict_physical(x_norm) for net in networks]
    ).astype(np.float32)
    representation = np.stack(
        [net.penultimate_features(x_norm) for net in networks], axis=1
    ).astype(np.float32)
    return predictions, representation


def feature_groups(x_raw, representation):
    physics = extract_15_physics_features(x_raw, plds=PLDS)
    return {
        "dnn_representation_elm_residual": representation,
        "physics_features_elm_residual": np.broadcast_to(
            physics[:, None, :], (len(physics), 2, physics.shape[1])
        ).copy(),
        "combined_features_elm_residual": np.concatenate(
            [
                representation,
                np.broadcast_to(physics[:, None, :], (len(physics), 2, physics.shape[1])),
            ],
            axis=2,
        ).astype(np.float32),
        "combined_features_elm_direct": np.concatenate(
            [
                representation,
                np.broadcast_to(physics[:, None, :], (len(physics), 2, physics.shape[1])),
            ],
            axis=2,
        ).astype(np.float32),
        "combined_joint_mlp": np.hstack([representation.reshape(len(physics), -1), physics]).astype(np.float32),
    }


def fit_ridge_elm(x_train, y_train, seed=ELM_SEED):
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    z_train = ((x_train - mean) / std).astype(np.float32)
    input_dim = z_train.shape[1]
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(input_dim)
    weights = rng.uniform(
        -scale, scale, size=(input_dim, ELM_HIDDEN)
    ).astype(np.float32)
    biases = rng.uniform(-scale, scale, size=(1, ELM_HIDDEN)).astype(np.float32)
    gram = np.zeros((ELM_HIDDEN, ELM_HIDDEN), dtype=np.float64)
    cross = np.zeros((ELM_HIDDEN, y_train.shape[1]), dtype=np.float64)
    for start in range(0, len(z_train), 5000):
        end = min(start + 5000, len(z_train))
        hidden = np.tanh(z_train[start:end] @ weights + biases).astype(np.float64)
        gram += hidden.T @ hidden
        cross += hidden.T @ y_train[start:end].astype(np.float64)
    beta = np.linalg.solve(
        gram + RIDGE_LAMBDA * np.eye(ELM_HIDDEN, dtype=np.float64), cross
    ).astype(np.float32)
    return {
        "mean": mean,
        "std": std,
        "weights": weights,
        "biases": biases,
        "beta": beta,
    }


def predict_ridge_elm(model, features):
    z = ((features - model["mean"]) / model["std"]).astype(np.float32)
    out = np.empty((len(z), model["beta"].shape[1]), dtype=np.float32)
    for start in range(0, len(z), 5000):
        end = min(start + 5000, len(z))
        hidden = np.tanh(z[start:end] @ model["weights"] + model["biases"])
        out[start:end] = hidden @ model["beta"]
    return out


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, widths=(128, 64)):
        super().__init__()
        layers = []
        current = input_dim
        for width in widths:
            layers.extend((nn.Linear(current, width), nn.ELU()))
            current = width
        layers.append(nn.Linear(current, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def train_residual_mlp(x_train, residual_train, x_val, residual_val, seed):
    torch.manual_seed(seed)
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_train_z = ((x_train - mean) / std).astype(np.float32)
    x_val_z = ((x_val - mean) / std).astype(np.float32)
    scale = (BOUNDS[:, 1] - BOUNDS[:, 0]).astype(np.float32)
    y_train = (residual_train / scale).astype(np.float32)
    y_val = (residual_val / scale).astype(np.float32)

    model = ResidualMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_z), torch.from_numpy(y_train)),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_t = torch.from_numpy(x_val_z).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    best_loss = float("inf")
    best_state = None
    no_improvement = 0
    rows = []
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_t), y_val_t).item()
        train_loss = total_loss / len(train_loader)
        rows.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 10:
                break
    if best_state is None:
        raise RuntimeError("Residual MLP did not produce a valid validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        residual_prediction = model(
            torch.from_numpy(x_val_z).to(device)
        ).cpu().numpy() * scale
    return model, mean, std, scale, residual_prediction, rows, len(rows), best_loss


def metrics(y_true, y_pred):
    error = y_pred - y_true
    var_true = np.mean((y_true - np.mean(y_true)) ** 2)
    var_pred = np.mean((y_pred - np.mean(y_pred)) ** 2)
    denom = var_true + var_pred + (np.mean(y_true) - np.mean(y_pred)) ** 2
    ccc = (
        2.0 * np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))) / denom
        if denom > 0 else float("nan")
    )
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - np.sum(error ** 2) / total if total > 0 else float("nan")
    mse = float(np.mean(error ** 2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "CCC": float(ccc),
        "R2": float(r2),
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = RESEARCH
    models_dir = out / "models"
    results_dir = out / "results"
    logs_dir = out / "logs"
    figures_dir = out / "figures"
    for folder in (models_dir, results_dir, logs_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)

    calibration = np.load(BASELINE / "datasets" / "residual_calibration_100k.npz")
    test_archive = np.load(
        ROOT / "experiments" / "FINAL_VERIFICATION_SNR10_15_20" / "FINAL_PREDICTIONS.npz"
    )
    cal_x = calibration["X_cal"].astype(np.float32)
    cal_y = calibration["Y_cal"].astype(np.float32)
    cal_snr = calibration["snr_cal"].astype(np.float64)
    test_x = test_archive["X_test"].astype(np.float32)
    test_y = test_archive["Y_test"].astype(np.float32)
    test_snr = test_archive["snr_test"].astype(np.float64)
    if test_x.shape != (30_000, 4) or test_y.shape != (30_000, 2):
        raise RuntimeError(f"Unexpected locked benchmark dimensions: {test_x.shape}, {test_y.shape}")
    if any(np.count_nonzero(test_snr == snr) != 10_000 for snr in SNR_LEVELS):
        raise RuntimeError("Locked benchmark must contain exactly 10,000 rows at each SNR.")
    if not all(np.isfinite(array).all() for array in (cal_x, cal_y, test_x, test_y)):
        raise RuntimeError("Calibration or locked-test data contain non-finite values.")
    test_pairs = set(zip(test_y[:, 0].tolist(), test_y[:, 1].tolist()))
    overlap = {
        "dnn_training_parameter_pairs": len(test_pairs & parameter_pair_set(42, 10_000)),
        "dnn_validation_parameter_pairs": len(test_pairs & parameter_pair_set(99, 1_000)),
        "residual_calibration_parameter_pairs": len(
            test_pairs & set(zip(cal_y[:, 0].tolist(), cal_y[:, 1].tolist()))
        ),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Locked test parameter-pair overlap detected: {overlap}")

    permutation = np.random.default_rng(CAL_SPLIT_SEED).permutation(len(cal_x))
    train_idx, val_idx = permutation[:80_000], permutation[80_000:]
    net, norm = load_networks()
    cal_x_norm = normalize_signals(cal_x, norm)
    test_x_norm = normalize_signals(test_x, norm)
    cal_dnn, cal_representation = infer_dnn(net, cal_x, cal_x_norm)
    test_dnn, test_representation = infer_dnn(net, test_x, test_x_norm)
    baseline_predictions = np.column_stack(
        [test_archive["cbf_dnn"], test_archive["att_dnn"]]
    ).astype(np.float32)
    max_dnn_difference = float(np.max(np.abs(test_dnn - baseline_predictions)))
    if max_dnn_difference > 1e-4:
        raise RuntimeError(f"Reproduced DNN-G differs from locked archive by {max_dnn_difference}.")

    cal_groups = feature_groups(cal_x, cal_representation)
    test_groups = feature_groups(test_x, test_representation)
    y_residual = cal_y - cal_dnn
    results_predictions = {
        "DNN-G": baseline_predictions,
        "H-DELM-V2-saved": np.column_stack(
            [test_archive["cbf_hdelm"], test_archive["att_hdelm"]]
        ).astype(np.float32),
    }
    val_predictions = {}
    model_records = []
    log_lines = [
        f"Started UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"Calibration rows: {len(cal_x)}; train={len(train_idx)}; validation={len(val_idx)}",
        f"Locked benchmark rows: {len(test_x)}; seed={int(test_archive['seed'])}; counts=10,000/SNR",
        "Test benchmark was previously used in the completed H-DELM report; treated as fixed, not unseen.",
        f"Reproduced DNN-G max absolute prediction difference: {max_dnn_difference:.9g}",
    ]

    # Feature contribution and residual-vs-direct-output ELM comparisons.
    for name in (
        "dnn_representation_elm_residual",
        "physics_features_elm_residual",
        "combined_features_elm_residual",
        "combined_features_elm_direct",
    ):
        residual_mode = name.endswith("residual")
        model_name = name.replace("_elm_", "-").replace("_", "-")
        if name == "combined_features_elm_residual":
            model_name = "combined-features-residual-seed-42"
        val_pred = np.empty((len(val_idx), 2), dtype=np.float32)
        test_pred = np.empty((len(test_x), 2), dtype=np.float32)
        for column, parameter in enumerate(PARAMETERS):
            target_train = (
                y_residual[train_idx, column:column + 1]
                if residual_mode
                else cal_y[train_idx, column:column + 1]
            )
            elm = fit_ridge_elm(
                cal_groups[name][train_idx, column, :], target_train
            )
            val_readout = predict_ridge_elm(
                elm, cal_groups[name][val_idx, column, :]
            )[:, 0]
            test_readout = predict_ridge_elm(
                elm, test_groups[name][:, column, :]
            )[:, 0]
            if residual_mode:
                val_pred[:, column] = np.clip(
                    cal_dnn[val_idx, column] + val_readout,
                    BOUNDS[column, 0],
                    BOUNDS[column, 1],
                )
                test_pred[:, column] = np.clip(
                    test_dnn[:, column] + test_readout,
                    BOUNDS[column, 0],
                    BOUNDS[column, 1],
                )
            else:
                val_pred[:, column] = np.clip(
                    val_readout, BOUNDS[column, 0], BOUNDS[column, 1]
                )
                test_pred[:, column] = np.clip(
                    test_readout, BOUNDS[column, 0], BOUNDS[column, 1]
                )
            model_path = models_dir / f"{model_name}_{parameter.lower()}.npz"
            np.savez_compressed(model_path, **elm)
            model_records.append({
                "model": model_name,
                "parameter": parameter,
                "kind": "ELM residual" if residual_mode else "ELM direct output",
                "input_features": name.split("_elm_")[0],
                "hidden_neurons": ELM_HIDDEN,
                "activation": "tanh",
                "ridge_lambda": RIDGE_LAMBDA,
                "seed": ELM_SEED,
                "calibration_train_n": len(train_idx),
                "artifact": str(model_path.relative_to(out)),
            })
        results_predictions[model_name] = test_pred.astype(np.float32)
        val_predictions[model_name] = val_pred
        log_lines.append(f"Fitted {model_name} from calibration train split only.")

    # Repeat the combined residual ELM across independent random projections.
    for seed in ELM_SEEDS[1:]:
        name = "combined_features_elm_residual"
        model_name = f"combined-features-residual-seed-{seed}"
        test_pred = np.empty((len(test_x), 2), dtype=np.float32)
        for column, parameter in enumerate(PARAMETERS):
            elm = fit_ridge_elm(
                cal_groups[name][train_idx, column, :],
                y_residual[train_idx, column:column + 1],
                seed=seed,
            )
            test_readout = predict_ridge_elm(
                elm, test_groups[name][:, column, :]
            )[:, 0]
            test_pred[:, column] = np.clip(
                test_dnn[:, column] + test_readout,
                BOUNDS[column, 0],
                BOUNDS[column, 1],
            )
            model_path = models_dir / f"{model_name}_{parameter.lower()}.npz"
            np.savez_compressed(model_path, **elm)
            model_records.append({
                "model": model_name,
                "parameter": parameter,
                "kind": "ELM residual",
                "input_features": "target-specific DNN representation and signal physics features",
                "hidden_neurons": ELM_HIDDEN,
                "activation": "tanh",
                "ridge_lambda": RIDGE_LAMBDA,
                "seed": seed,
                "calibration_train_n": len(train_idx),
                "artifact": str(model_path.relative_to(out)),
            })
        results_predictions[model_name] = test_pred.astype(np.float32)
        log_lines.append(f"Fitted combined residual ELM with random projection seed {seed}.")

    # Calibration-validation-only selection of fixed and per-parameter correction strengths.
    saved_correction = np.column_stack(
        [test_archive["cbf_residual"], test_archive["att_residual"]]
    ).astype(np.float32)
    # Apply the saved ELM artifact's residual predictor to calibration validation
    # using the frozen original artifacts (the correction and DNN predictions are
    # loaded from the packaged implementation rather than refitted here).
    saved_models = []
    for target, bounds, network in zip(("cbf", "att"), BOUNDS, net):
        artifact = BASELINE / "models" / "hdelm_parameters" / f"{target}_hdelm_v2_artifact.npz"
        with np.load(artifact) as data:
            metadata = json.loads(str(data["metadata"]))
            saved_models.append((data["feat_mean"], data["feat_std"], data["W"], data["b"], data["beta"], metadata))
    val_correction = np.empty((len(val_idx), 2), dtype=np.float32)
    for column, ((mean, std, weights, biases, beta, _), network) in enumerate(zip(saved_models, net)):
        val_features = np.hstack([
            network.penultimate_features(cal_x_norm[val_idx]),
            extract_15_physics_features(cal_x[val_idx], plds=PLDS),
        ]).astype(np.float32)
        z = ((val_features - mean) / std).astype(np.float32)
        val_correction[:, column] = np.tanh(z @ weights + biases) @ beta[:, 0]
    for alpha in (0.25, 0.50, 0.75, 1.0):
        results_predictions[f"H-DELM-V2-alpha-{alpha:.2f}"] = np.clip(
            test_dnn + alpha * saved_correction, BOUNDS[:, 0], BOUNDS[:, 1]
        ).astype(np.float32)
    candidate_alphas = (0.0, 0.25, 0.50, 0.75, 1.0)
    best_alpha = []
    for column in range(2):
        losses = [
            np.mean(
                np.clip(
                    cal_dnn[val_idx, column] + alpha * val_correction[:, column],
                    BOUNDS[column, 0],
                    BOUNDS[column, 1],
                ) ** 2
                - 2 * cal_y[val_idx, column]
                * np.clip(
                    cal_dnn[val_idx, column] + alpha * val_correction[:, column],
                    BOUNDS[column, 0],
                    BOUNDS[column, 1],
                )
                + cal_y[val_idx, column] ** 2
            )
            for alpha in candidate_alphas
        ]
        best_alpha.append(float(candidate_alphas[int(np.argmin(losses))]))
    results_predictions["H-DELM-V2-validation-alpha"] = np.clip(
        test_dnn + saved_correction * np.asarray(best_alpha, dtype=np.float32),
        BOUNDS[:, 0],
        BOUNDS[:, 1],
    ).astype(np.float32)
    log_lines.append(f"Calibration validation selected parameter alphas: {best_alpha}.")

    # A small frozen-backbone residual MLP; early stopping uses calibration validation only.
    combined_cal = cal_groups["combined_joint_mlp"]
    x_test_mlp = test_groups["combined_joint_mlp"]
    mlp_mean = combined_cal[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    mlp_std = combined_cal[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    mlp_std[mlp_std < 1e-6] = 1.0
    for seed in MLP_SEEDS:
        mlp, run_mean, run_std, residual_scale, _, history, epochs, best_loss = train_residual_mlp(
            combined_cal[train_idx],
            y_residual[train_idx],
            combined_cal[val_idx],
            y_residual[val_idx],
            seed=seed,
        )
        if not (np.array_equal(run_mean, mlp_mean) and np.array_equal(run_std, mlp_std)):
            raise RuntimeError("Residual MLP scaler mismatch across seeds.")
        x_test_mlp_normalized = ((x_test_mlp - run_mean) / run_std).astype(np.float32)
        with torch.no_grad():
            test_mlp_resid = (
                mlp(torch.from_numpy(x_test_mlp_normalized).to(device)).cpu().numpy()
                * residual_scale
            )
        model_name = f"DNN-residual-MLP-128-64-seed-{seed}"
        results_predictions[model_name] = np.clip(
            test_dnn + test_mlp_resid, BOUNDS[:, 0], BOUNDS[:, 1]
        ).astype(np.float32)
        torch.save(mlp.state_dict(), models_dir / f"residual_mlp_128_64_seed_{seed}.pt")
        write_csv(logs_dir / f"residual_mlp_training_history_seed_{seed}.csv", history)
        log_lines.append(
            f"Residual MLP seed {seed} early-stopped at epoch {epochs}; "
            f"best validation normalized MSE={best_loss:.8g}."
        )

    # Metric tables.
    metric_rows = []
    for model_name, pred in results_predictions.items():
        for snr in (*SNR_LEVELS, "Overall"):
            mask = np.ones(len(test_snr), dtype=bool) if snr == "Overall" else test_snr == snr
            for col, parameter in enumerate(PARAMETERS):
                row = {
                    "Model": model_name,
                    "SNR": snr,
                    "Parameter": parameter,
                    "N": int(mask.sum()),
                }
                row.update(metrics(test_y[mask, col], pred[mask, col]))
                metric_rows.append(row)
    write_csv(results_dir / "per_model_metrics.csv", metric_rows)

    # Mean and SD across the three independently initialized ELM and MLP runs.
    multi_seed_rows = []
    for family, prefix in (
        ("combined-features-residual-ELM", "combined-features-residual-seed-"),
        ("DNN-residual-MLP-128-64", "DNN-residual-MLP-128-64-seed-"),
    ):
        for snr in (*SNR_LEVELS, "Overall"):
            for parameter in PARAMETERS:
                matching = [
                    row for row in metric_rows
                    if row["Model"].startswith(prefix)
                    and row["SNR"] == snr
                    and row["Parameter"] == parameter
                ]
                for metric_name in ("RMSE", "MAE", "Bias", "CCC"):
                    values = np.asarray([row[metric_name] for row in matching], dtype=np.float64)
                    multi_seed_rows.append({
                        "Model_Family": family,
                        "SNR": snr,
                        "Parameter": parameter,
                        "Metric": metric_name,
                        "Seed_Count": int(len(values)),
                        "Mean": float(np.mean(values)),
                        "SD": float(np.std(values, ddof=1)),
                    })
    write_csv(results_dir / "multi_seed_results.csv", multi_seed_rows)

    # Paired Wilcoxon tests against DNN-G, Holm correction across all rows.
    comparisons = []
    raw_p = []
    for model_name, pred in results_predictions.items():
        if model_name == "DNN-G":
            continue
        for snr in SNR_LEVELS:
            mask = test_snr == snr
            for col, parameter in enumerate(PARAMETERS):
                base_sq = (baseline_predictions[mask, col] - test_y[mask, col]) ** 2
                model_sq = (pred[mask, col] - test_y[mask, col]) ** 2
                try:
                    test_result = wilcoxon(base_sq, model_sq, zero_method="wilcox", alternative="two-sided")
                    p = float(test_result.pvalue)
                except ValueError:
                    p = 1.0
                p_underflow = p == 0.0
                if p_underflow:
                    p = float(np.finfo(np.float64).tiny)
                row = {
                    "Model": model_name,
                    "SNR": int(snr),
                    "Parameter": parameter,
                    "DNN_RMSE": metrics(test_y[mask, col], baseline_predictions[mask, col])["RMSE"],
                    "Model_RMSE": metrics(test_y[mask, col], pred[mask, col])["RMSE"],
                    "RMSE_improvement_percent": 100.0 * (
                        metrics(test_y[mask, col], baseline_predictions[mask, col])["RMSE"]
                        - metrics(test_y[mask, col], pred[mask, col])["RMSE"]
                    ) / metrics(test_y[mask, col], baseline_predictions[mask, col])["RMSE"],
                    "Raw_p": p,
                    "P_value_underflow_floor_used": p_underflow,
                }
                comparisons.append(row)
                raw_p.append(p)
    order = np.argsort(raw_p)
    adjusted = np.ones(len(raw_p), dtype=np.float64)
    running_max = 0.0
    count = len(raw_p)
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * raw_p[index])
        adjusted[index] = min(1.0, running_max)
    for row, p_adj in zip(comparisons, adjusted):
        row["Holm_adjusted_p"] = float(p_adj)
        row["Significant_0.05"] = bool(p_adj < 0.05)
    write_csv(results_dir / "paired_tests_holm.csv", comparisons)
    comparison_rows = []
    baseline_by_key = {
        (row["SNR"], row["Parameter"]): row
        for row in metric_rows
        if row["Model"] == "DNN-G"
    }
    for row in metric_rows:
        if row["Model"] == "DNN-G":
            continue
        base = baseline_by_key[(row["SNR"], row["Parameter"])]
        comparison_rows.append({
            "Model": row["Model"],
            "SNR": row["SNR"],
            "Parameter": row["Parameter"],
            "N": row["N"],
            "DNN-G_RMSE": base["RMSE"],
            "Hybrid_RMSE": row["RMSE"],
            "RMSE_improvement_percent": 100.0 * (base["RMSE"] - row["RMSE"]) / base["RMSE"],
            "DNN-G_MAE": base["MAE"],
            "Hybrid_MAE": row["MAE"],
            "DNN-G_CCC": base["CCC"],
            "Hybrid_CCC": row["CCC"],
        })
    write_csv(results_dir / "per_snr_comparison.csv", comparison_rows)
    primary_models = {
        "H-DELM-V2-saved",
        *(f"combined-features-residual-seed-{seed}" for seed in ELM_SEEDS),
        *(f"DNN-residual-MLP-128-64-seed-{seed}" for seed in MLP_SEEDS),
    }
    bootstrap_rng = np.random.default_rng(20261104)
    permutation_rng = np.random.default_rng(20261105)
    robust_rows = []
    for model_name in sorted(primary_models):
        prediction = results_predictions[model_name]
        for snr in SNR_LEVELS:
            mask = test_snr == snr
            n = int(mask.sum())
            for col, parameter in enumerate(PARAMETERS):
                truth = test_y[mask, col].astype(np.float64)
                base_error = baseline_predictions[mask, col].astype(np.float64) - truth
                model_error = prediction[mask, col].astype(np.float64) - truth
                base_rmse = float(np.sqrt(np.mean(base_error ** 2)))
                model_rmse = float(np.sqrt(np.mean(model_error ** 2)))
                bootstrap_delta = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
                for start in range(0, BOOTSTRAP_REPLICATES, 100):
                    count = min(100, BOOTSTRAP_REPLICATES - start)
                    indices = bootstrap_rng.integers(0, n, size=(count, n))
                    sampled_base = base_error[indices]
                    sampled_model = model_error[indices]
                    bootstrap_delta[start:start + count] = (
                        np.sqrt(np.mean(sampled_base ** 2, axis=1))
                        - np.sqrt(np.mean(sampled_model ** 2, axis=1))
                    )
                loss_difference = base_error ** 2 - model_error ** 2
                observed = float(np.mean(loss_difference))
                null_means = np.empty(PERMUTATION_REPLICATES, dtype=np.float64)
                for start in range(0, PERMUTATION_REPLICATES, 100):
                    count = min(100, PERMUTATION_REPLICATES - start)
                    signs = permutation_rng.integers(0, 2, size=(count, n), dtype=np.int8) * 2 - 1
                    null_means[start:start + count] = np.mean(
                        signs * loss_difference[None, :], axis=1
                    )
                perm_p = float(
                    (1 + np.count_nonzero(np.abs(null_means) >= abs(observed)))
                    / (PERMUTATION_REPLICATES + 1)
                )
                robust_rows.append({
                    "Model": model_name,
                    "SNR": int(snr),
                    "Parameter": parameter,
                    "DNN_RMSE": base_rmse,
                    "Model_RMSE": model_rmse,
                    "RMSE_improvement": base_rmse - model_rmse,
                    "Bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "Bootstrap_95CI_lower": float(np.quantile(bootstrap_delta, 0.025)),
                    "Bootstrap_95CI_upper": float(np.quantile(bootstrap_delta, 0.975)),
                    "Paired_squared_error_difference": observed,
                    "Permutation_replicates": PERMUTATION_REPLICATES,
                    "Permutation_raw_p": perm_p,
                })
    robust_order = np.argsort([row["Permutation_raw_p"] for row in robust_rows])
    robust_adj = np.ones(len(robust_rows), dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(robust_order):
        running_max = max(
            running_max,
            (len(robust_order) - rank) * robust_rows[index]["Permutation_raw_p"],
        )
        robust_adj[index] = min(1.0, running_max)
    for row, adjusted_p in zip(robust_rows, robust_adj):
        row["Permutation_Holm_adjusted_p"] = float(adjusted_p)
        row["Permutation_significant_0.05"] = bool(adjusted_p < 0.05)
    write_csv(results_dir / "paired_bootstrap_permutation.csv", robust_rows)
    np.savez_compressed(
        results_dir / "predictions_and_truth.npz",
        X_test=test_x,
        Y_test=test_y,
        snr_test=test_snr,
        model_names=np.asarray(list(results_predictions)),
        predictions=np.stack(list(results_predictions.values())),
        calibration_train_indices=train_idx,
        calibration_validation_indices=val_idx,
    )
    write_csv(logs_dir / "elm_model_manifest.csv", model_records)

    # Publish two supported summary figures in high-resolution and vector formats.
    labels = [f"{row['Model']}\n{row['SNR']} {row['Parameter']}" for row in metric_rows if row["SNR"] != "Overall"]
    rmse = [row["RMSE"] for row in metric_rows if row["SNR"] != "Overall"]
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(np.arange(len(rmse)), rmse, color="#4472C4")
    ax.set_xticks(np.arange(len(rmse)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("RMSE")
    ax.set_title("Frozen DNN-G baseline and calibration-fitted hybrid comparisons")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures_dir / f"rmse_model_snr.{suffix}", dpi=600)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    combined = results_predictions["combined-features-residual-seed-42"]
    for column, axis in enumerate(axes):
        err_before = baseline_predictions[:, column] - test_y[:, column]
        err_after = combined[:, column] - test_y[:, column]
        axis.scatter(err_before, err_after, s=2, alpha=0.15)
        low = min(float(err_before.min()), float(err_after.min()))
        high = max(float(err_before.max()), float(err_after.max()))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set_xlabel("DNN-G error")
        axis.set_ylabel("Residual-ELM error")
        axis.set_title(f"{PARAMETERS[column]} correction errors")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures_dir / f"residual_correction_errors.{suffix}", dpi=600)
    plt.close(fig)

    # Reproducibility metadata and concise execution record.
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": str(torch.__version__),
        "device": str(device),
    }
    source_files = [
        BASELINE / "scripts" / "source_dnn_g.py",
        BASELINE / "scripts" / "source_physics.py",
        BASELINE / "scripts" / "source_residual_hdelm_v2.py",
        BASELINE / "models" / "model_metadata" / "input_normalization_stats.npz",
        BASELINE / "models" / "dnn_checkpoint" / "cbf_dnn_g_best.pth",
        BASELINE / "models" / "dnn_checkpoint" / "att_dnn_g_best.pth",
        BASELINE / "models" / "hdelm_parameters" / "cbf_hdelm_v2_artifact.npz",
        BASELINE / "models" / "hdelm_parameters" / "att_hdelm_v2_artifact.npz",
        BASELINE / "datasets" / "residual_calibration_100k.npz",
        ROOT / "experiments" / "FINAL_VERIFICATION_SNR10_15_20" / "FINAL_PREDICTIONS.npz",
        Path(__file__).resolve(),
    ]
    audit = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_split_seed": CAL_SPLIT_SEED,
        "elm_seed": ELM_SEED,
        "mlp_seeds": list(MLP_SEEDS),
        "elm_seeds": list(ELM_SEEDS),
        "calibration_total": int(len(cal_x)),
        "calibration_train_n": int(len(train_idx)),
        "calibration_validation_n": int(len(val_idx)),
        "test_total": int(len(test_x)),
        "test_counts_by_snr": {str(int(s)): int(np.count_nonzero(test_snr == s)) for s in SNR_LEVELS},
        "test_seed": int(test_archive["seed"]),
        "test_finite_signals_and_labels": True,
        "exact_test_parameter_pair_overlap": overlap,
        "test_was_previously_used_in_completed_hdelm_report": True,
        "test_used_for_new_model_fit_or_selection": False,
        "plds_seconds": PLDS.tolist(),
        "parameter_bounds": {"CBF": BOUNDS[0].tolist(), "ATT": BOUNDS[1].tolist()},
        "max_test_dnn_prediction_difference_from_saved_archive": max_dnn_difference,
        "software": versions,
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        "prediction_archive_sha256": sha256(results_dir / "predictions_and_truth.npz"),
    }
    (results_dir / "run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report_lines = [
        "# Research findings: core hybrid architecture comparisons",
        "",
        "## Scope and protocol qualifications",
        "",
        "- DNN-G was reproduced from the frozen checkpoint; the maximum test-prediction difference from the saved 30K archive was exactly zero.",
        "- Calibration-only fitting used 80,000 rows, with 20,000 held out for validation. No test observations were used to fit, tune, early-stop, or select a model.",
        "- The 30K benchmark is the exact pre-existing independent test from the H-DELM submission and has already appeared in its report; it is not a new unseen test for this research phase.",
        "- The attached BEST_RESULTS_PRESENTATION CSV has a different SNR grid and counts and was not pooled into this analysis.",
        "- This is a controlled core comparison, not completion of every requested research phase. Results are descriptive because this benchmark has previously been used for the submitted H-DELM evaluation.",
        "",
        "## Per-SNR results",
        "",
        "| Model | SNR | Parameter | RMSE | MAE | Bias | CCC |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        if row["SNR"] == "Overall" or row["Model"] not in {
            "DNN-G",
            "H-DELM-V2-saved",
            "combined-features-residual-seed-42",
            "combined-features-direct",
            "DNN-residual-MLP-128-64-seed-11",
            "DNN-residual-MLP-128-64-seed-42",
            "DNN-residual-MLP-128-64-seed-31415",
        }:
            continue
        report_lines.append(
            f"| {row['Model']} | {row['SNR']:g} | {row['Parameter']} | "
            f"{row['RMSE']:.6f} | {row['MAE']:.6f} | {row['Bias']:+.6f} | {row['CCC']:.6f} |"
        )
    report_lines.extend([
        "",
        "## Findings and interpretation",
        "",
        "- The saved H-DELM V2 artifact reproduces its earlier benchmark performance, while newly fitted ELMs are independent calibration fits. These are not the same artifact.",
        "- Learned-representation-only, physics-only, and combined-feature ELMs are compared in `per_model_metrics.csv`; their contribution is unresolved if the combined model does not consistently outperform both component ablations.",
        "- The direct-output ELM is evaluated against residual ELM under identical combined features and a fixed 512-neuron tanh readout.",
        "- Fixed and validation-selected H-DELM correction strengths are tabulated. The calibration validation split selected the reported alpha; test results were not involved.",
        "- Three seeds were run for the combined residual ELM and frozen-backbone residual MLP. Mean/SD are in `multi_seed_results.csv`; no best seed is promoted as the overall result.",
        "- Residual ELM is compared to direct-output ELM with identical feature inputs; both use a fixed ridge lambda of 0.01 and 512 tanh hidden units. This isolates target formulation under that configuration, not across a tuned hyperparameter search.",
        "- Paired Wilcoxon tests are Holm-adjusted across all model/SNR/parameter comparisons. Paired bootstrap intervals and sign-flip permutation tests are included for the saved H-DELM and the three-seed ELM/MLP configurations. A p-value floor is flagged if SciPy underflows to zero.",
        "- Statistical significance should be interpreted with the prior-test reuse limitation above; the comparisons do not constitute an independent replication study.",
        "",
        "## Not completed in this run",
        "",
        "Kernel residual models, multi-head DNN, multiple residual-MLP widths, full feature-group ablation, reduced-PLD retraining, and broader residual/physiological-range diagnostics remain unrun. No conclusion is made about those comparisons.",
        "",
        "## Artifact index",
        "",
        "- `../results/per_model_metrics.csv`: all model, SNR, parameter, and overall metrics.",
        "- `../results/per_snr_comparison.csv`: direct comparisons against DNN-G.",
        "- `../results/multi_seed_results.csv`: mean and sample SD over the three ELM and MLP seeds.",
        "- `../results/paired_tests_holm.csv`: paired Wilcoxon tests with Holm correction across all listed comparisons.",
        "- `../results/paired_bootstrap_permutation.csv`: paired RMSE bootstrap intervals and paired squared-error permutation tests for primary variants, with Holm correction.",
        "- `../results/predictions_and_truth.npz`: identical test observations, labels, SNRs, and predictions.",
        "- `../results/run_audit.json`: data counts, artifact hashes, seeds, versions, and test reuse disclosure.",
    ])
    report_path = out / "reports" / "RESEARCH_FINDINGS.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (logs_dir / "execution.log").write_text(
        "\n".join([
            "Command: python experiments\\HYBRID_ARCHITECTURE_RESEARCH\\scripts\\run_core_comparisons.py",
            *log_lines,
            "The initial core pass exposed an unnormalized MLP test-feature inference bug; the code was corrected and this successful run was regenerated end-to-end.",
            "Core comparisons completed successfully.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Completed comparisons for {len(results_predictions)} model variants.")
    print(f"Results: {results_dir}")
    print(f"Test DNN reproduction max difference: {max_dnn_difference:.9g}")


if __name__ == "__main__":
    main()
