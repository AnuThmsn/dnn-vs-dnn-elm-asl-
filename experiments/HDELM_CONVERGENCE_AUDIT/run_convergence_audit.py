"""Audit original H-DELM DNN convergence histories and create annotated plots."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "experiments" / "HDELM_CONVERGENCE_AUDIT"
EXP = ROOT / "experiments" / "paper_reproduction_snr10_15_20"
SOURCE = EXP / "models" / "dnn_g.py"
PIPELINE = EXP / "run_reproduction_pipeline.py"
PLOT_SOURCE = ROOT / "submited h-delm" / "scripts" / "publication_utils.py"
STATS = ROOT / "submited h-delm" / "datasets" / "dataset_statistics.csv"
PACKAGE_MODELS = ROOT / "submited h-delm" / "models"
ORIGINAL_HISTORY_DIR = AUDIT / "original_histories"
FIGURES = AUDIT / "figures"
TITLE = "DNN Backbone Training and Validation Convergence (H-DELM)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_history(name: str) -> pd.DataFrame:
    path = ORIGINAL_HISTORY_DIR / name
    frame = pd.read_csv(path)
    expected = {"epoch", "train_loss", "val_loss", "val_rmse_physical"}
    if set(frame.columns) != expected:
        raise RuntimeError(f"Unexpected history columns for {path.name}: {frame.columns}")
    if not np.array_equal(frame["epoch"].to_numpy(), np.arange(1, len(frame) + 1)):
        raise RuntimeError(f"Non-consecutive epochs in {path.name}")
    if not np.isfinite(frame[list(expected)].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Non-finite history values in {path.name}")
    return frame


def checkpoint_audit(target: str, bounds: tuple[float, float]) -> dict:
    spec = importlib.util.spec_from_file_location("audited_dnn_g", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DNN source module at {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.DNN_G_Net(
        input_dim=4,
        n_hidden_layers=4,
        n_neurons=64,
        y_min=bounds[0],
        y_max=bounds[1],
    )
    checkpoint = EXP / "models" / f"{target}_dnn_g_best.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    forbidden = [
        name for name, child in model.named_modules()
        if isinstance(child, (nn.Dropout, nn.modules.batchnorm._BatchNorm))
    ]
    if forbidden:
        raise RuntimeError(f"Unexpected stochastic/normalization layers: {forbidden}")
    packaged_checkpoint = PACKAGE_MODELS / "dnn_checkpoint" / checkpoint.name
    if sha256(checkpoint) != sha256(packaged_checkpoint):
        raise RuntimeError(f"Source and packaged checkpoints differ for {target}.")
    history_name = f"{target}_dnn_g_training_history.csv"
    source_history = EXP / "results" / history_name
    packaged_history = PACKAGE_MODELS / "model_metadata" / history_name
    preserved_history = ORIGINAL_HISTORY_DIR / history_name
    history_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (source_history, packaged_history, preserved_history)
    }
    if len(set(history_hashes.values())) != 1:
        raise RuntimeError(f"Original histories do not match for {target}: {history_hashes}")
    return {
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_modified_utc": datetime.fromtimestamp(
            checkpoint.stat().st_mtime
        ).astimezone().isoformat(),
        "checkpoint_is_plain_state_dict": all(isinstance(k, str) for k in state),
        "state_dict_entries": len(state),
        "packaged_checkpoint_path": str(packaged_checkpoint.relative_to(ROOT)),
        "packaged_checkpoint_sha256": sha256(packaged_checkpoint),
        "history_sha256": history_hashes,
        "history_modified_utc": datetime.fromtimestamp(
            source_history.stat().st_mtime
        ).astimezone().isoformat(),
        "history_and_checkpoint_modified_time_delta_seconds": abs(
            source_history.stat().st_mtime - checkpoint.stat().st_mtime
        ),
        "strict_checkpoint_load": True,
        "dropout_or_batchnorm_modules": forbidden,
    }


def summarize(target: str, label: str, frame: pd.DataFrame, info: dict) -> dict:
    best_idx = int(frame["val_loss"].idxmin())
    best_epoch = int(frame.loc[best_idx, "epoch"])
    non_improvement_epochs = len(frame) - best_epoch
    patience = 10
    if non_improvement_epochs != patience:
        raise RuntimeError(
            f"{target} early-stop count {non_improvement_epochs} did not match patience {patience}."
        )
    first = frame.iloc[0]
    last = frame.iloc[-1]
    best = frame.iloc[best_idx]
    return {
        "target": target,
        "label": label,
        "epochs_completed": int(len(frame)),
        "first_train_loss": float(first.train_loss),
        "first_val_loss": float(first.val_loss),
        "last_train_loss": float(last.train_loss),
        "last_val_loss": float(last.val_loss),
        "best_val_epoch": best_epoch,
        "best_val_loss": float(best.val_loss),
        "train_loss_at_best_val_epoch": float(best.train_loss),
        "train_minus_val_at_best_epoch": float(best.train_loss - best.val_loss),
        "validation_lower_than_training_epochs": int(
            np.count_nonzero(frame["val_loss"].to_numpy() < frame["train_loss"].to_numpy())
        ),
        "epochs_validation_lower_than_training_fraction": float(
            np.mean(frame["val_loss"].to_numpy() < frame["train_loss"].to_numpy())
        ),
        "training_loss_drop_percent": float(
            100 * (first.train_loss - last.train_loss) / first.train_loss
        ),
        "best_to_last_validation_increase_percent": float(
            100 * (last.val_loss - best.val_loss) / best.val_loss
        ),
        "early_stopping_patience": patience,
        "non_improving_epochs_after_best": non_improvement_epochs,
        "checkpoint_claim": (
            "Saved checkpoint is the best-validation state by the training source; "
            "the checkpoint itself stores only a state_dict, not an epoch field."
        ),
        **info,
    }


def plot_history(frame: pd.DataFrame, label: str, output_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        frame["epoch"],
        frame["train_loss"],
        color="#276FBF",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Training: mean batch MSE during updates",
    )
    ax.plot(
        frame["epoch"],
        frame["val_loss"],
        color="#D1495B",
        linewidth=1.8,
        marker="s",
        markersize=3,
        label="Validation: epoch-end mean batch MSE",
    )
    best = frame.loc[frame["val_loss"].idxmin()]
    ax.scatter(
        [best["epoch"]],
        [best["val_loss"]],
        color="#D1495B",
        edgecolor="black",
        linewidth=0.6,
        s=55,
        zorder=5,
        label=f"Best validation: epoch {int(best['epoch'])}",
    )
    ax.set_title(f"{TITLE}\n{label}", fontsize=13, pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Normalized MSE", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.text(
        0.01,
        -0.20,
        "Losses are equal-weighted batch means: final batches contain 64/512 train and 160/512 validation samples. "
        "No dropout or batch normalization is present.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"{output_stem}.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def combined_plot(cbf: pd.DataFrame, att: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=False)
    for ax, frame, label in zip(axes, (cbf, att), ("CBF", "ATT")):
        ax.plot(
            frame["epoch"], frame["train_loss"], color="#276FBF",
            marker="o", markersize=3, linewidth=1.8,
            label="Training: mean batch MSE during updates",
        )
        ax.plot(
            frame["epoch"], frame["val_loss"], color="#D1495B",
            marker="s", markersize=3, linewidth=1.8,
            label="Validation: epoch-end mean batch MSE",
        )
        best = frame.loc[frame["val_loss"].idxmin()]
        ax.scatter(
            [best["epoch"]], [best["val_loss"]], color="#D1495B",
            edgecolor="black", linewidth=0.6, s=55, zorder=5,
            label=f"Best validation: epoch {int(best['epoch'])}",
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Normalized MSE", fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.tick_params(axis="both", labelsize=8.5)
        ax.grid(True, axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(TITLE, fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.025,
        "Training values average batch means during updates; validation uses epoch-end eval weights. "
        "Incomplete final batches (64 train, 160 validation) are equally weighted as full batches; no dropout/BatchNorm.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"combined_cbf_att_convergence.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_report(summaries: list[dict], dataset_rows: list[dict]) -> None:
    lookup = {row["target"]: row for row in summaries}
    cbf = lookup["cbf"]
    att = lookup["att"]
    dataset_summary_lines = []
    for split in ("DNN training", "DNN validation"):
        for variable in ("CBF", "ATT", "SNR"):
            record = next(
                row for row in dataset_rows
                if row["Split"] == split and row["Variable"] == variable
            )
            if variable == "SNR":
                description = record["Units/distribution"]
                dataset_summary_lines.append(
                    f"| {split} | SNR | {int(record['N']):,} | "
                    f"{record['Minimum']:.1f}–{record['Maximum']:.1f} | — | — | {description} |"
                )
            else:
                dataset_summary_lines.append(
                    f"| {split} | {variable} | {int(record['N']):,} | "
                    f"{record['Minimum']:.6f}–{record['Maximum']:.6f} | "
                    f"{record['Mean']:.6f} | {record['SD']:.6f} | {record['Units/distribution']} |"
                )
    report = f"""# H-DELM DNN convergence audit

## Scope and preservation

This audit is isolated under `experiments/HDELM_CONVERGENCE_AUDIT/`. It does
not modify `submited h-delm/` and does not retrain either network. The original
CSV histories were copied byte-for-byte before analysis to
`original_histories/`; their SHA-256 values match both the source experiment
history files and the packaged copies. Figures and this report are newly
generated under this audit directory only.

## Original loss implementation

Source: `experiments/paper_reproduction_snr10_15_20/models/dnn_g.py`,
`train_dnn_g`, lines 94–172.

- Both targets use the same normalized target formula on train and validation:
  `y_norm = (y_phys - y_min) / (y_max - y_min)`. The respective physical bounds
  are shared between the two splits; CBF is `[0,100]`, ATT is `[0.5,3.0]`.
- Both predictions use `sigmoid(logit)` and the same `nn.MSELoss()` (default
  mean reduction) on scalar output batches. Target dimensions are both
  `(N,1)` after `unsqueeze(1)`. There is no target scaling mismatch between
  training and validation.
- The optimizer loop calls `net.train()` for training minibatches, computes
  each batch loss, calls backward/step, and accumulates the scalar batch means.
  At epoch end, it divides the sum by the number of batches. This is an
  **unweighted mean of batch means**, not the exact sample-weighted epoch mean
  when the final batch is incomplete. With batch size 512, the 1,000,000-row training
  set uses 1,954 batches (last batch 64); the 100,000-row validation set uses
  196 batches (last batch 160). Thus a sample in the final batch is weighted
  8× (train) or 3.2× (validation) as much as a sample in a full batch in the
  reported epoch average.
- Validation then calls `net.eval()` and evaluates the model under
  `torch.no_grad()` after all epoch optimizer steps. The saved `val_loss`
  therefore uses the final weights of that epoch.
- The model has no dropout or batch-normalization modules. `train()` vs
  `eval()` does not activate a stochastic/normalization layer for this DNN.

The submitted combined convergence figure is generated by
`submited h-delm/scripts/publication_utils.py`, `generate_figures`, lines
296–308. It reads the packaged CBF/ATT history CSVs and plots their
`train_loss` and `val_loss` columns directly. Those CSV files were verified
byte-identical to the source experiment histories preserved in this audit.
The source plotting code and finalized package were read-only during this
audit.

The pipeline uses `generate_paper_dataset(n_unique=10000, n_snr_levels=100,
seed=42)` for training and `n_unique=1000`, `seed=99` for validation
(`run_reproduction_pipeline.py`, lines 40–62 and 84–85). It sends the complete
cohorts to each independent target network. Counts are 1,000,000 train and
100,000 validation observations for CBF and likewise for ATT. Inputs are
standardized from training input statistics only. Recorded dataset summaries
are in `submited h-delm/datasets/dataset_statistics.csv`.

| Split | Variable | N | Min–max | Mean | SD | Units/distribution |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(dataset_summary_lines)}

## Numerical history findings

Values below are the original, unaltered history values (normalized MSE).

| Target | Epochs | Train loss, epoch 1 → last | Validation loss, epoch 1 → last | Best validation epoch/loss | Train−validation at best epoch | Epochs validation < training |
|---|---:|---:|---:|---:|---:|---:|
| CBF | {cbf['epochs_completed']} | {cbf['first_train_loss']:.9f} → {cbf['last_train_loss']:.9f} | {cbf['first_val_loss']:.9f} → {cbf['last_val_loss']:.9f} | {cbf['best_val_epoch']} / {cbf['best_val_loss']:.9f} | {cbf['train_minus_val_at_best_epoch']:.9f} | {cbf['validation_lower_than_training_epochs']} / {cbf['epochs_completed']} |
| ATT | {att['epochs_completed']} | {att['first_train_loss']:.9f} → {att['last_train_loss']:.9f} | {att['first_val_loss']:.9f} → {att['last_val_loss']:.9f} | {att['best_val_epoch']} / {att['best_val_loss']:.9f} | {att['train_minus_val_at_best_epoch']:.9f} | {att['validation_lower_than_training_epochs']} / {att['epochs_completed']} |

### ATT validation-below-training behavior

The ATT validation loss is lower than the recorded training loss in
{att['validation_lower_than_training_epochs']} of {att['epochs_completed']}
epochs. At the final recorded epoch, train MSE is
{att['last_train_loss']:.9f} and validation MSE is {att['last_val_loss']:.9f}
(validation is {(att['last_train_loss']-att['last_val_loss'])/att['last_train_loss']*100:.2f}% lower). This does **not** result from
dropout, batch normalization, different target normalization, or a different
loss function: those are absent or shared as described above.

The two cohorts are independently generated with seeds 42 and 99 and are not
identical samples. Their recorded parameter summaries show training ATT mean
{next(x['Mean'] for x in dataset_rows if x['Split']=='DNN training' and x['Variable']=='ATT'):.6f}s, SD
{next(x['SD'] for x in dataset_rows if x['Split']=='DNN training' and x['Variable']=='ATT'):.6f}s versus validation ATT mean
{next(x['Mean'] for x in dataset_rows if x['Split']=='DNN validation' and x['Variable']=='ATT'):.6f}s, SD
{next(x['SD'] for x in dataset_rows if x['Split']=='DNN validation' and x['Variable']=='ATT'):.6f}s. Training contains 10,000 unique
physiological parameter pairs repeated across 100 SNR levels; validation has
1,000 different pairs repeated across the same 100 SNR levels. Sampling
variation and these cohort differences can make validation easier. The loss
history alone cannot attribute the gap quantitatively to those factors.

Two implementation details can contribute to the ATT gap without indicating a
mode or normalization bug. First, the training curve is an online epoch
average collected as weights change from batch to batch, while validation
uses the final epoch weights. Second, equal weighting of the smaller final
batch overweights it; validation is not shuffled, and dataset generation
orders each unique parameter pair in a consecutive block of 100 SNR
observations. Its final 160 rows therefore represent only the final two ATT
parameter pairs (100 rows from the last pair, 60 from the preceding pair),
with the same aggregate weight as each 512-row batch. Training shuffles rows,
so its last 64 are a random subset instead. These provide implementation-based
reasons for a validation estimate that can be lower, but the available
per-sample losses are insufficient to quantify their contribution to the
8.06% ATT gap. These curves are valid records of the implemented optimization
and validation procedure, but they are **not strictly matched epoch-end,
sample-weighted train-vs-validation measurements**.

## CBF convergence and checkpoint selection

CBF training loss falls by {cbf['training_loss_drop_percent']:.2f}% from epoch 1
to the final epoch, while validation loss reaches its minimum at epoch
{cbf['best_val_epoch']} and ends
{cbf['best_to_last_validation_increase_percent']:.3f}% above that minimum.
Early stopping is implemented correctly: `best_state` is updated only when
validation loss strictly improves, `no_imp` increments otherwise, and the
loop stops at 10 consecutive non-improving epochs. The CBF run completes 37
epochs, with the best at 27 and 10 later non-improving epochs. ATT completes
26 epochs, best at 16, followed by 10 non-improving epochs.

The training function restores `best_state` before returning (source line 170).
The pipeline then saves that returned model's state dictionary (CBF pipeline
lines 117–123; ATT lines 131–137). Thus the checkpoint is the best-validation
state, not the last-epoch state. The saved checkpoint is a plain state
dictionary with no embedded epoch field, so the exact epoch attribution is
supported by the training/pipeline code and matching history/checkpoint
artifacts rather than checkpoint metadata. Source and packaged histories and
checkpoints are byte-identical; their file timestamps are recorded in
`history_checkpoint_audit.json`.

## Recalculation limits and issues

The original 1M training and 100K validation arrays were generated in memory
and were not persisted. Only summary statistics, histories, and checkpoints
remain. Therefore it is not possible to recalculate evaluation-mode train and
validation losses from the saved checkpoint on the original cohorts, or to
reconstruct comparable post-epoch curves, without regenerating replacement
cohorts. This audit does not substitute regenerated data, retrain, or alter
loss values.

The original recorded loss calculations are mathematically valid as
batch-averaged training-time and epoch-end validation metrics using the same
normalized MSE definition. However, they are not exact sample-weighted epoch
MSEs because the incomplete final batches are averaged with the same
weight as complete batches, and the training curve uses predictions made
throughout optimizer updates rather than a fixed final-epoch checkpoint. No
evidence indicates an incorrect target dimension, reduction within a batch,
target scale, early stopping, or mode switch.

## Figure files

- `figures/cbf_convergence.png`
- `figures/cbf_convergence.pdf`
- `figures/att_convergence.png`
- `figures/att_convergence.pdf`
- `figures/combined_cbf_att_convergence.png`
- `figures/combined_cbf_att_convergence.pdf`

All figures plot original history values without smoothing, interpolation, or
manual adjustment. The training/validation aggregation distinction,
incomplete-batch weighting, and absence of dropout/batch normalization are
disclosed in the figure notes. The ATT validation-below-training pattern is
plausible from cohort differences, tail-batch weighting, and online training
loss; the available artifacts cannot establish a more specific causal
decomposition.

## Recommendation for the paper

**Replace the original convergence graph with the audited, clearly annotated
figures** (or omit the convergence plot if the methodology cannot be explained).
The original curves are not fabricated, but their aggregation gives unequal
per-sample weights to partial batches and the train curve is not measured at
fixed end-of-epoch weights. Keep the original values and explain these
limitations; do not describe the ATT pattern as evidence of leakage or a
training-mode regularization effect.
"""
    (AUDIT / "CONVERGENCE_AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    history_files = {
        "cbf": ("cbf_dnn_g_training_history.csv", "CBF", (0.0, 100.0)),
        "att": ("att_dnn_g_training_history.csv", "ATT", (0.5, 3.0)),
    }
    summaries = []
    frames = {}
    for target, (filename, label, bounds) in history_files.items():
        frame = read_history(filename)
        info = checkpoint_audit(target, bounds)
        summary = summarize(target, label, frame, info)
        frames[target] = frame
        summaries.append(summary)
        plot_history(frame, label, f"{target}_convergence")

    combined_plot(frames["cbf"], frames["att"])
    with (AUDIT / "loss_history_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    dataset_rows = pd.read_csv(STATS).to_dict(orient="records")
    source_manifest = {
        "training_source": str(SOURCE.relative_to(ROOT)),
        "pipeline_source": str(PIPELINE.relative_to(ROOT)),
        "training_source_sha256": sha256(SOURCE),
        "pipeline_source_sha256": sha256(PIPELINE),
        "original_plot_source": str(PLOT_SOURCE.relative_to(ROOT)),
        "original_plot_source_sha256": sha256(PLOT_SOURCE),
        "dataset_statistics_sha256": sha256(STATS),
        "original_history_preserved_before_plot_generation": True,
        "training_arrays_persisted": False,
        "validation_arrays_persisted": False,
        "retrained": False,
        "model_architecture_or_parameters_changed": False,
        "run_command": "python experiments\\HDELM_CONVERGENCE_AUDIT\\run_convergence_audit.py",
        "checkpoints": {row["target"]: {
            "path": row["checkpoint_path"],
            "sha256": row["checkpoint_sha256"],
            "packaged_path": row["packaged_checkpoint_path"],
            "packaged_sha256": row["packaged_checkpoint_sha256"],
            "strict_load": row["strict_checkpoint_load"],
            "best_validation_epoch_from_history": row["best_val_epoch"],
            "checkpoint_embeds_epoch": False,
            "history_sha256_by_copy": row["history_sha256"],
        } for row in summaries},
    }
    (AUDIT / "history_checkpoint_audit.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )
    write_report(summaries, dataset_rows)
    print("Convergence audit complete.")
    for row in summaries:
        print(
            f"{row['label']}: best validation epoch {row['best_val_epoch']}, "
            f"{row['epochs_completed']} epochs."
        )


if __name__ == "__main__":
    main()
