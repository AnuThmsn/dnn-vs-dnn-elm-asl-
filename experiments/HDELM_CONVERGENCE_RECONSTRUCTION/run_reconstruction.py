"""Reconstruct H-DELM DNN datasets and produce comparable convergence curves."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "HDELM_CONVERGENCE_RECONSTRUCTION"
EXP = ROOT / "experiments" / "paper_reproduction_snr10_15_20"
SOURCE_MODELS = EXP / "models"
PACKAGE_MODELS = ROOT / "submited h-delm" / "models"
AUDIT = ROOT / "experiments" / "HDELM_CONVERGENCE_AUDIT"

DATA_SEEDS = {"train": 42, "validation": 99}
CONTROLLED_SEED = 42
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
SNR_COUNT = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_source_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_dataset(physics, n_unique: int, seed: int):
    rng = np.random.default_rng(seed)
    cbf_unique = rng.uniform(physics.CBF_MIN, physics.CBF_MAX, n_unique).astype(np.float64)
    att_unique = rng.uniform(physics.ATT_MIN, physics.ATT_MAX, n_unique).astype(np.float64)
    clean_unique = physics.compute_physical_signals(
        physics.PLDS, physics.TAU, cbf_unique, att_unique
    )
    snr_levels = np.linspace(10.0, 20.0, SNR_COUNT, dtype=np.float64)
    cbf_all = np.repeat(cbf_unique, SNR_COUNT)
    att_all = np.repeat(att_unique, SNR_COUNT)
    clean_all = np.repeat(clean_unique, SNR_COUNT, axis=0)
    snr_all = np.tile(snr_levels, n_unique)
    x_raw = physics.add_rician_noise(clean_all, snr_all, rng)
    y_phys = np.stack([cbf_all, att_all], axis=1).astype(np.float32)
    return x_raw, y_phys, snr_all


def normalize_targets(y_phys: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    return ((y_phys - y_min) / (y_max - y_min)).astype(np.float32)


def evaluation_mse(
    model: nn.Module,
    x: np.ndarray,
    y_norm: np.ndarray,
    batch_size: int = 8192,
) -> float:
    model.eval()
    squared_error_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end]).to(DEVICE)
            yb = torch.from_numpy(y_norm[start:end, None]).to(DEVICE)
            prediction = model.forward_normalized(xb)
            if prediction.shape != yb.shape:
                raise RuntimeError(
                    f"Prediction/target shape mismatch: {prediction.shape} vs {yb.shape}"
                )
            squared_error_sum += float(torch.sum((prediction - yb) ** 2).item())
            sample_count += end - start
    if sample_count != len(x):
        raise RuntimeError(f"Evaluated {sample_count} of {len(x)} samples.")
    return squared_error_sum / sample_count


def eval_saved_checkpoint(
    target: str,
    label: str,
    bounds: tuple[float, float],
    architecture,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    checkpoint = SOURCE_MODELS / f"{target}_dnn_g_best.pth"
    packaged = PACKAGE_MODELS / "dnn_checkpoint" / checkpoint.name
    checkpoint_hash_before = sha256_file(checkpoint)
    packaged_hash = sha256_file(packaged)
    if checkpoint_hash_before != packaged_hash:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint} vs {packaged}")
    model = architecture.DNN_G_Net(
        input_dim=4,
        n_hidden_layers=4,
        n_neurons=64,
        y_min=bounds[0],
        y_max=bounds[1],
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE, weights_only=True), strict=True
    )
    train_mse = evaluation_mse(
        model, x_train, normalize_targets(y_train, *bounds)
    )
    val_mse = evaluation_mse(
        model, x_val, normalize_targets(y_val, *bounds)
    )
    checkpoint_hash_after = sha256_file(checkpoint)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise RuntimeError(f"Existing checkpoint changed during evaluation: {checkpoint}")
    return {
        "target": label,
        "model_source": "existing_saved_best_checkpoint",
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "training_mse_normalized": train_mse,
        "validation_mse_normalized": val_mse,
        "training_rmse_physical": float(np.sqrt(train_mse) * (bounds[1] - bounds[0])),
        "validation_rmse_physical": float(np.sqrt(val_mse) * (bounds[1] - bounds[0])),
        "validation_minus_training_mse": val_mse - train_mse,
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash_before,
        "packaged_checkpoint_sha256": packaged_hash,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
    }


def train_controlled(
    target: str,
    label: str,
    bounds: tuple[float, float],
    architecture,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[list[dict], dict]:
    # Match the original pipeline's reset-before-each-target behavior while
    # also seeding Python/NumPy and the independent DataLoader generator.
    seed_everything(CONTROLLED_SEED)
    model = architecture.DNN_G_Net(
        input_dim=4,
        n_hidden_layers=4,
        n_neurons=64,
        y_min=bounds[0],
        y_max=bounds[1],
    ).to(DEVICE)
    y_train_norm = normalize_targets(y_train, *bounds)
    y_val_norm = normalize_targets(y_val, *bounds)
    train_data = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train_norm[:, None])
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(CONTROLLED_SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        pin_memory=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction="mean")
    history = []
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_normalized(xb)
            loss = criterion(prediction, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Both datasets are evaluated after all optimizer steps for this epoch.
        train_mse = evaluation_mse(model, x_train, y_train_norm)
        val_mse = evaluation_mse(model, x_val, y_val_norm)
        model.train()
        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        early_stop = stale_epochs >= PATIENCE
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "training_mse_normalized": train_mse,
            "validation_mse_normalized": val_mse,
            "training_rmse_physical": float(np.sqrt(train_mse) * (bounds[1] - bounds[0])),
            "validation_rmse_physical": float(np.sqrt(val_mse) * (bounds[1] - bounds[0])),
            "learning_rate": current_lr,
            "best_validation_epoch_so_far": best_epoch,
            "best_validation_mse_so_far": best_val,
            "epochs_without_improvement": stale_epochs,
            "improved_validation": improved,
            "early_stopping_triggered": early_stop,
        })
        if epoch == 1 or epoch % 5 == 0 or early_stop:
            print(
                f"{label} epoch {epoch}: train={train_mse:.8f}, "
                f"val={val_mse:.8f}, best={best_epoch}",
                flush=True,
            )
        if early_stop:
            break
    if best_state is None:
        raise RuntimeError(f"No best checkpoint was recorded for {label}.")

    # Store only in the new reconstruction directory.
    model.load_state_dict(best_state, strict=True)
    checkpoint_path = OUT / "models" / f"{target}_controlled_best.pth"
    torch.save(best_state, checkpoint_path)
    metadata = {
        "target": label,
        "seed": CONTROLLED_SEED,
        "best_validation_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_mse_normalized": best_val,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "architecture_source": str((EXP / "models" / "dnn_g.py").relative_to(ROOT)),
        "architecture": {
            "input_features": 4,
            "hidden_layers": 4,
            "neurons_per_hidden_layer": 64,
            "activation": "ELU",
            "bounded_output": "sigmoid normalized to target bounds",
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "gradient_clip_norm": GRAD_CLIP,
            "loss": "mean squared error on normalized target",
            "train_loss_reporting": "epoch-end model.eval() sample-weighted MSE over all train samples",
            "validation_loss_reporting": "epoch-end model.eval() sample-weighted MSE over all validation samples",
        },
    }
    return history, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(histories: dict[str, list[dict]]) -> None:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    title = "DNN Backbone Training and Validation Convergence (H-DELM)"
    settings = [
        ("CBF", histories["cbf"], "cbf_corrected_convergence"),
        ("ATT", histories["att"], "att_corrected_convergence"),
    ]
    for label, rows, stem in settings:
        frame = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(
            frame["epoch"], frame["training_mse_normalized"],
            color="#276FBF", linewidth=1.8, label="Training MSE (eval mode)",
        )
        ax.plot(
            frame["epoch"], frame["validation_mse_normalized"],
            color="#D1495B", linewidth=1.8, label="Validation MSE (eval mode)",
        )
        best_epoch = int(frame.loc[frame["validation_mse_normalized"].idxmin(), "epoch"])
        best_row = frame.loc[frame["epoch"] == best_epoch].iloc[0]
        ax.scatter(
            [best_epoch], [best_row["validation_mse_normalized"]],
            color="#D1495B", edgecolor="black", s=48, zorder=5,
            label=f"Best validation: epoch {best_epoch}",
        )
        ax.set_title(f"{title}\n{label}", fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Normalized MSE", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / f"{stem}.png", dpi=600, bbox_inches="tight")
        fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (label, rows, _) in zip(axes, settings):
        frame = pd.DataFrame(rows)
        ax.plot(
            frame["epoch"], frame["training_mse_normalized"],
            color="#276FBF", linewidth=1.8, label="Training MSE (eval mode)",
        )
        ax.plot(
            frame["epoch"], frame["validation_mse_normalized"],
            color="#D1495B", linewidth=1.8, label="Validation MSE (eval mode)",
        )
        best_epoch = int(frame.loc[frame["validation_mse_normalized"].idxmin(), "epoch"])
        best_loss = float(frame["validation_mse_normalized"].min())
        ax.scatter(
            [best_epoch], [best_loss], color="#D1495B",
            edgecolor="black", s=48, zorder=5,
            label=f"Best validation: epoch {best_epoch}",
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Normalized MSE", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures / "combined_corrected_convergence.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures / "combined_corrected_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def preserve_historical_figures() -> list[str]:
    destination = OUT / "figures" / "historical_original_audit"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in (
        "cbf_convergence.png", "cbf_convergence.pdf",
        "att_convergence.png", "att_convergence.pdf",
        "combined_cbf_att_convergence.png", "combined_cbf_att_convergence.pdf",
    ):
        source = AUDIT / "figures" / name
        if not source.exists():
            raise FileNotFoundError(f"Original audit figure missing: {source}")
        target = destination / name
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(ROOT)))
    return copied


def build_report(
    saved_results: list[dict],
    controlled_metadata: dict[str, dict],
    controlled_histories: dict[str, list[dict]],
    metadata: dict,
) -> None:
    audit_summary = pd.read_csv(AUDIT / "loss_history_summary.csv")
    original_by_target = {row["target"]: row for row in audit_summary.to_dict("records")}
    saved_by_target = {row["target"].lower(): row for row in saved_results}
    controlled_by_target = {
        target: rows for target, rows in controlled_histories.items()
    }
    sections = []
    for target, label in (("cbf", "CBF"), ("att", "ATT")):
        saved = saved_by_target[label.lower()]
        original = original_by_target[target]
        rows = controlled_by_target[target]
        best_row = min(rows, key=lambda row: row["validation_mse_normalized"])
        final_row = rows[-1]
        controlled_below_count = sum(
            row["validation_mse_normalized"] < row["training_mse_normalized"]
            for row in rows
        )
        sections.append(f"""### {label}

| Evidence source | Train normalized MSE | Validation normalized MSE | Validation below train? |
|---|---:|---:|---|
| Original best checkpoint evaluated consistently on reconstructed cohorts | {saved['training_mse_normalized']:.9f} | {saved['validation_mse_normalized']:.9f} | {"Yes" if saved['validation_mse_normalized'] < saved['training_mse_normalized'] else "No"} |
| Controlled run at best-validation epoch {best_row['epoch']} | {best_row['training_mse_normalized']:.9f} | {best_row['validation_mse_normalized']:.9f} | {"Yes" if best_row['validation_mse_normalized'] < best_row['training_mse_normalized'] else "No"} |
| Controlled run at final epoch {final_row['epoch']} | {final_row['training_mse_normalized']:.9f} | {final_row['validation_mse_normalized']:.9f} | {"Yes" if final_row['validation_mse_normalized'] < final_row['training_mse_normalized'] else "No"} |

- Original history best validation epoch: {original['best_val_epoch']}; controlled
  best epoch: {best_row['epoch']}; controlled epochs completed:
  {len(rows)} (early stop triggered: {rows[-1]['early_stopping_triggered']}).
- Original historical training loss at its last epoch was
  {original['last_train_loss']:.9f}; original validation loss was
  {original['last_val_loss']:.9f}. Those original training losses were collected
  during updates and are not directly comparable to epoch-end eval losses.
- Controlled validation was below training in {controlled_below_count} of
  {len(rows)} epochs. Physical-unit RMSE at the best controlled epoch:
  train {best_row['training_rmse_physical']:.6f}; validation
  {best_row['validation_rmse_physical']:.6f}.
""")
    report = f"""# H-DELM convergence reconstruction report

## Scope and safeguards

All new outputs are confined to `experiments/HDELM_CONVERGENCE_RECONSTRUCTION/`.
The original `HDELM_CONVERGENCE_AUDIT/` is preserved, and its original
historical plots were copied to `figures/historical_original_audit/`. The
finalized `submited h-delm/` package, its histories, and existing checkpoints
were not modified. Checkpoint SHA-256 values were checked before and after
evaluation/training and against packaged checkpoint copies. Controlled-run
checkpoints exist only in this reconstruction directory.

## Original audit findings

The initial audit found that historical training loss averages minibatch mean
losses while model weights change, whereas validation loss is measured in
evaluation mode after the epoch updates. Both use the same normalized-MSE
definition, but the history aggregation gives a smaller final batch the same
weight as a full batch. No dropout or batch normalization is present. Best
historical validation epochs were CBF 27 and ATT 16. The original arrays were
not persisted, so the exact historical losses cannot be recomputed on those
arrays.

## Dataset reconstruction

The generator was reconstructed directly from
`experiments/paper_reproduction_snr10_15_20/run_reproduction_pipeline.py`
(`generate_paper_dataset`) and `experiments/physics.py`, without modifying
either source:

- Training: 10,000 unique CBF/ATT pairs × 100 linearly spaced SNRs from 10
  through 20, generated using seed 42 = 1,000,000 rows.
- Validation: 1,000 unique pairs × the same 100 SNR levels, seed 99 =
  100,000 rows.
- Original PLDs, Buxton GKM, dual-channel quadrature Rician noise, and
  physiological parameter bounds were used.
- Input mean and standard deviation were recalculated from reconstructed
  training signals only, then applied to both cohorts.

The data arrays, normalization statistics, metadata, shapes, seeds, and
SHA-256 hashes are in `datasets/`. These are **reconstructed deterministic
cohorts**, not byte-identical historical arrays: the original training and
validation arrays were not saved, and numerical/library/device differences
may also affect generated floating-point values.

## Existing checkpoint evaluation

The original CBF/ATT checkpoints were loaded strictly and evaluated with
`model.eval()` and `torch.no_grad()` on all reconstructed observations.
Normalized MSE was accumulated as total squared error divided by total sample
count; no gradient updates occurred. Target normalization used the same
physical bounds for train and validation.

{chr(10).join(sections)}

Existing-checkpoint loss values and hashes are in
`results/saved_checkpoint_comparable_losses.csv`. The ATT validation/train
relationship for this reconstructed dataset is therefore the result shown
above; it need not match the historical online-averaged curves. A difference
does not imply that either the historical checkpoint or this evaluation is
incorrect.

## Controlled convergence reconstruction

CBF and ATT were trained separately using the source `DNN_G_Net` architecture:
4 hidden layers × 64 units, ELU activations, sigmoid-bounded target output,
Adam at 1e-3, batch size 512, gradient clipping at 1.0, patience 10, and
normalized MSE. Python, NumPy, CPU/CUDA PyTorch seeds were set to
{CONTROLLED_SEED} before each target network initialization. Data-generation
seeds remain 42 and 99. Library versions, device, deterministic backend
settings, source hashes, and checkpoint hashes are recorded in
`datasets/reconstruction_metadata.json` and `models/`.

After each epoch's updates, both datasets were evaluated in `model.eval()`
with `torch.no_grad()`. Batch squared-error sums were accumulated and divided
by the exact number of samples, giving sample-weighted MSE. Validation was
never used for gradient updates. Early stopping used the complete validation
MSE; best controlled checkpoints and best epochs are recorded in the
controlled histories and metadata.

The controlled run does not reproduce the original optimizer trajectory:
the historical training arrays were unavailable, and a new seeded
reconstruction run was performed. It is not expected to reproduce the
original best epochs exactly. Besides the non-persisted historical arrays and
the changed loss reporting (epoch-end evaluation instead of online minibatch
loss), the controlled run gives the shuffled DataLoader its own generator
seeded to 42. The original creates the shuffled DataLoader after model
initialization and uses the global PyTorch generator, so minibatch order is
not the same even though the target initialization seed is 42. GPU/library
kernel nondeterminism may also contribute. Architecture and hyperparameters
were not changed; the controlled histories are not asserted to be bitwise
repeatable across hardware/software environments.

## Figures and recommendation

The corrected curves plot controlled-run sample-weighted, eval-mode losses:

- `figures/cbf_corrected_convergence.png` and `.pdf`
- `figures/att_corrected_convergence.png` and `.pdf`
- `figures/combined_corrected_convergence.png` and `.pdf`

Original audit plots remain separately available under
`figures/historical_original_audit/`. The corrected controlled curves are the
appropriate figures to present when describing sample-weighted, end-of-epoch
train and validation losses. Label them as a **controlled reconstruction**,
not the exact historical training trajectory. If the publication specifically
reports the original fitted model, retain its historical curves only with an
explicit caveat about online/unweighted-batch aggregation and missing original
arrays; do not silently substitute the reconstruction for historical
training results.

## Artifacts and reproduction

See `README.md` for the reproduction command. Detailed arrays and hashes are
in `datasets/`; frozen-checkpoint metrics and new per-epoch histories are in
`results/`; controlled best checkpoints are in `models/`.
"""
    (OUT / "CONVERGENCE_RECONSTRUCTION_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    for folder in ("datasets", "models", "results", "figures"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    physics = load_source_module("reconstruction_physics", ROOT / "experiments" / "physics.py")
    architecture = load_source_module(
        "reconstruction_dnn_g", EXP / "models" / "dnn_g.py"
    )
    pipeline_path = EXP / "run_reproduction_pipeline.py"
    config_path = EXP / "config" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    source_files = [
        ROOT / "experiments" / "physics.py",
        EXP / "run_reproduction_pipeline.py",
        EXP / "models" / "dnn_g.py",
        config_path,
    ]
    existing_checkpoint_paths = {
        target: SOURCE_MODELS / f"{target}_dnn_g_best.pth"
        for target in ("cbf", "att")
    }
    checkpoint_hashes_before = {
        target: sha256_file(path)
        for target, path in existing_checkpoint_paths.items()
    }
    historical_paths = {
        target: AUDIT / "original_histories" / f"{target}_dnn_g_training_history.csv"
        for target in ("cbf", "att")
    }
    historical_hashes_before = {
        target: sha256_file(path) for target, path in historical_paths.items()
    }
    package_manifest_before = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (ROOT / "submited h-delm").rglob("*")
        if path.is_file()
    }

    print("Reconstructing 1,000,000-row training cohort (seed 42)...", flush=True)
    x_train_raw, y_train, snr_train = generate_dataset(physics, 10_000, DATA_SEEDS["train"])
    print("Reconstructing 100,000-row validation cohort (seed 99)...", flush=True)
    x_val_raw, y_val, snr_val = generate_dataset(physics, 1_000, DATA_SEEDS["validation"])
    if x_train_raw.shape != (1_000_000, 4) or y_train.shape != (1_000_000, 2):
        raise RuntimeError(f"Unexpected training shapes: {x_train_raw.shape}, {y_train.shape}")
    if x_val_raw.shape != (100_000, 4) or y_val.shape != (100_000, 2):
        raise RuntimeError(f"Unexpected validation shapes: {x_val_raw.shape}, {y_val.shape}")
    if not all(np.isfinite(array).all() for array in (x_train_raw, y_train, x_val_raw, y_val)):
        raise RuntimeError("Reconstructed datasets contain non-finite values.")

    x_mean = x_train_raw.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = x_train_raw.std(axis=0, keepdims=True).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    x_train = ((x_train_raw - x_mean) / x_std).astype(np.float32)
    x_val = ((x_val_raw - x_mean) / x_std).astype(np.float32)
    dataset_dir = OUT / "datasets"
    np.savez_compressed(
        dataset_dir / "reconstructed_training_dataset.npz",
        X_raw=x_train_raw,
        X_normalized=x_train,
        Y_physical=y_train,
        snr=snr_train,
        seed=np.asarray(DATA_SEEDS["train"]),
    )
    np.savez_compressed(
        dataset_dir / "reconstructed_validation_dataset.npz",
        X_raw=x_val_raw,
        X_normalized=x_val,
        Y_physical=y_val,
        snr=snr_val,
        seed=np.asarray(DATA_SEEDS["validation"]),
    )
    np.savez_compressed(
        dataset_dir / "input_normalization_train_only.npz",
        X_mean=x_mean,
        X_std=x_std,
    )
    dataset_hashes = {
        "training_dataset_file_sha256": sha256_file(
            dataset_dir / "reconstructed_training_dataset.npz"
        ),
        "validation_dataset_file_sha256": sha256_file(
            dataset_dir / "reconstructed_validation_dataset.npz"
        ),
        "normalization_file_sha256": sha256_file(
            dataset_dir / "input_normalization_train_only.npz"
        ),
        "training_arrays": {
            name: sha256_array(array)
            for name, array in (
                ("X_raw", x_train_raw), ("X_normalized", x_train),
                ("Y_physical", y_train), ("snr", snr_train),
            )
        },
        "validation_arrays": {
            name: sha256_array(array)
            for name, array in (
                ("X_raw", x_val_raw), ("X_normalized", x_val),
                ("Y_physical", y_val), ("snr", snr_val),
            )
        },
        "normalization_arrays": {
            "X_mean": sha256_array(x_mean),
            "X_std": sha256_array(x_std),
        },
    }

    bounds = {
        "cbf": (physics.CBF_MIN, physics.CBF_MAX),
        "att": (physics.ATT_MIN, physics.ATT_MAX),
    }
    saved_results = []
    for target, label in (("cbf", "CBF"), ("att", "ATT")):
        saved_results.append(eval_saved_checkpoint(
            target, label, bounds[target], architecture,
            x_train, y_train[:, 0 if target == "cbf" else 1],
            x_val, y_val[:, 0 if target == "cbf" else 1],
        ))
    write_csv(OUT / "results" / "saved_checkpoint_comparable_losses.csv", saved_results)

    controlled_histories = {}
    controlled_metadata = {}
    for target, label in (("cbf", "CBF"), ("att", "ATT")):
        column = 0 if target == "cbf" else 1
        print(f"Controlled sample-weighted training: {label}...", flush=True)
        history, model_metadata = train_controlled(
            target, label, bounds[target], architecture,
            x_train, y_train[:, column], x_val, y_val[:, column],
        )
        controlled_histories[target] = history
        controlled_metadata[target] = model_metadata
        write_csv(
            OUT / "results" / f"{target}_controlled_sample_weighted_history.csv",
            history,
        )
        (OUT / "models" / f"{target}_controlled_metadata.json").write_text(
            json.dumps(model_metadata, indent=2), encoding="utf-8"
        )

    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_files
    }
    historical_copies = preserve_historical_figures()
    plot_convergence(controlled_histories)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pytorch": str(torch.__version__),
        "matplotlib": matplotlib.__version__,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "reconstruction_not_historical_byte_identity": True,
        "historical_training_arrays_persisted": False,
        "historical_validation_arrays_persisted": False,
        "dataset_seeds": DATA_SEEDS,
        "controlled_torch_python_numpy_seed": CONTROLLED_SEED,
        "dataloader_seed": CONTROLLED_SEED,
        "sample_counts": {"training": len(x_train), "validation": len(x_val)},
        "array_shapes": {
            "X_train_raw": list(x_train_raw.shape),
            "X_train_normalized": list(x_train.shape),
            "Y_train_physical": list(y_train.shape),
            "X_validation_raw": list(x_val_raw.shape),
            "X_validation_normalized": list(x_val.shape),
            "Y_validation_physical": list(y_val.shape),
        },
        "plds_seconds": physics.PLDS.tolist(),
        "snr_levels": np.linspace(10.0, 20.0, SNR_COUNT).tolist(),
        "parameters": {
            "CBF": [physics.CBF_MIN, physics.CBF_MAX],
            "ATT": [physics.ATT_MIN, physics.ATT_MAX],
        },
        "noise_model": "source experiments.physics.add_rician_noise (dual-channel quadrature Rician magnitude)",
        "simulator": "source experiments.physics.compute_physical_signals (Buxton GKM)",
        "generation_procedure_source_sha256": source_hashes,
        "reconstructed_dataset_sha256": dataset_hashes,
        "library_versions": versions,
        "configuration": config,
        "historical_plot_copies": historical_copies,
        "existing_checkpoint_hashes_before": checkpoint_hashes_before,
        "original_audit_history_hashes_before": historical_hashes_before,
        "finalized_package_file_hashes_before": package_manifest_before,
        "controlled_run": controlled_metadata,
    }
    (dataset_dir / "reconstruction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    build_report(saved_results, controlled_metadata, controlled_histories, metadata)

    checkpoint_hashes_after = {
        target: sha256_file(path)
        for target, path in existing_checkpoint_paths.items()
    }
    historical_hashes_after = {
        target: sha256_file(path) for target, path in historical_paths.items()
    }
    package_manifest_after = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (ROOT / "submited h-delm").rglob("*")
        if path.is_file()
    }
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("An existing source checkpoint changed.")
    if historical_hashes_before != historical_hashes_after:
        raise RuntimeError("An original audit history changed.")
    if package_manifest_before != package_manifest_after:
        raise RuntimeError("A file in the finalized submission package changed.")
    print("Reconstruction completed; frozen checkpoints, histories, and submission package verified unchanged.")


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _resolve_device()


if __name__ == "__main__":
    main()
