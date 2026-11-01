"""Reproduce the independent SNR test using the packaged frozen artifacts."""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from source_physics import (  # noqa: E402
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    compute_physical_signals, add_rician_noise,
)
from source_dnn_g import DNN_G_Net, device  # noqa: E402
from source_residual_hdelm_v2 import ResidualHDELMV2  # noqa: E402

SEED = 20261101
SNR_LEVELS = (10.0, 15.0, 20.0)
N_PER_SNR = 10_000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    rng = np.random.default_rng(SEED)
    x_chunks, y_chunks, snr_chunks = [], [], []
    for level in SNR_LEVELS:
        cbf = rng.uniform(CBF_MIN, CBF_MAX, N_PER_SNR).astype(np.float64)
        att = rng.uniform(ATT_MIN, ATT_MAX, N_PER_SNR).astype(np.float64)
        clean = compute_physical_signals(PLDS, TAU, cbf, att)
        noisy = add_rician_noise(clean, level, rng)
        x_chunks.append(noisy)
        y_chunks.append(np.column_stack([cbf, att]).astype(np.float32))
        snr_chunks.append(np.full(N_PER_SNR, level, dtype=np.float64))

    x_test = np.concatenate(x_chunks).astype(np.float32)
    y_test = np.concatenate(y_chunks).astype(np.float32)
    snr_test = np.concatenate(snr_chunks)
    if x_test.shape != (30_000, 4) or y_test.shape != (30_000, 2):
        raise RuntimeError(f"Unexpected test shape: X={x_test.shape}, Y={y_test.shape}")

    def parameter_pairs(seed, unique_count):
        source_rng = np.random.default_rng(seed)
        cbf_values = source_rng.uniform(CBF_MIN, CBF_MAX, unique_count).astype(np.float32)
        att_values = source_rng.uniform(ATT_MIN, ATT_MAX, unique_count).astype(np.float32)
        return set(zip(cbf_values.tolist(), att_values.tolist()))

    test_pairs = set(zip(y_test[:, 0].tolist(), y_test[:, 1].tolist()))
    calibration = np.load(ROOT / "datasets" / "residual_calibration_100k.npz")
    calibration_pairs = set(zip(calibration["Y_cal"][:, 0].tolist(), calibration["Y_cal"][:, 1].tolist()))
    intersections = {
        "train_exact_parameter_pair_intersections": len(test_pairs & parameter_pairs(42, 10_000)),
        "validation_exact_parameter_pair_intersections": len(test_pairs & parameter_pairs(99, 1_000)),
        "calibration_exact_parameter_pair_intersections": len(test_pairs & calibration_pairs),
    }
    if any(intersections.values()):
        raise RuntimeError(f"Test parameter-pair overlap detected: {intersections}")

    model_dir = ROOT / "models"
    ckpt_dir = model_dir / "dnn_checkpoint"
    norm = np.load(model_dir / "model_metadata" / "input_normalization_stats.npz")
    x_normalized = ((x_test - norm["X_mean"]) / norm["X_std"]).astype(np.float32)

    cbf_net = DNN_G_Net(4, 4, 64, CBF_MIN, CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(ckpt_dir / "cbf_dnn_g_best.pth", map_location=device, weights_only=True))
    cbf_net.eval()
    att_net = DNN_G_Net(4, 4, 64, ATT_MIN, ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(ckpt_dir / "att_dnn_g_best.pth", map_location=device, weights_only=True))
    att_net.eval()

    cbf_elm = ResidualHDELMV2.load_artifact(
        str(model_dir / "hdelm_parameters" / "cbf_hdelm_v2_artifact.npz"), dnn_net=cbf_net
    )
    att_elm = ResidualHDELMV2.load_artifact(
        str(model_dir / "hdelm_parameters" / "att_hdelm_v2_artifact.npz"), dnn_net=att_net
    )
    cbf_final, cbf_dnn, cbf_resid = cbf_elm.predict_physical(x_test, x_normalized)
    att_final, att_dnn, att_resid = att_elm.predict_physical(x_test, x_normalized)

    prediction_path = ROOT / "results" / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        X_test=x_test,
        Y_test=y_test,
        snr_test=snr_test,
        cbf_dnn=cbf_dnn,
        att_dnn=att_dnn,
        cbf_hdelm=cbf_final,
        att_hdelm=att_final,
        cbf_residual=cbf_resid,
        att_residual=att_resid,
        seed=np.asarray(SEED),
        plds=PLDS,
    )
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "training_seed": 42,
        "validation_seed": 99,
        "calibration_seed": int(calibration["seed"]),
        "samples_per_snr": {str(int(level)): int(np.count_nonzero(snr_test == level)) for level in SNR_LEVELS},
        "total_samples": int(len(x_test)),
        "input_shape": list(x_test.shape),
        "target_shape": list(y_test.shape),
        "plds_seconds": PLDS.tolist(),
        "cbf_range": [CBF_MIN, CBF_MAX],
        "att_range_seconds": [ATT_MIN, ATT_MAX],
        "simulator": "source_physics.compute_physical_signals and add_rician_noise",
        "noise_model": "dual-channel quadrature Rician magnitude noise",
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": str(torch.__version__),
        },
        "finite_signals_and_labels": bool(np.isfinite(x_test).all() and np.isfinite(y_test).all()),
        "exact_test_label_pair_overlap": intersections,
        "models": ["frozen DNN-G checkpoints", "saved H-DELM V2 ELM artifacts"],
        "test_used_for_fit_or_tuning": False,
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ckpt_dir / "cbf_dnn_g_best.pth",
                ckpt_dir / "att_dnn_g_best.pth",
                model_dir / "model_metadata" / "input_normalization_stats.npz",
                model_dir / "hdelm_parameters" / "cbf_hdelm_v2_artifact.npz",
                model_dir / "hdelm_parameters" / "att_hdelm_v2_artifact.npz",
                ROOT / "datasets" / "residual_calibration_100k.npz",
            )
        },
        "prediction_file_sha256": sha256(prediction_path),
    }
    (ROOT / "results" / "test_data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Evaluated {len(x_test):,} independent observations; wrote {prediction_path}")


if __name__ == "__main__":
    main()
