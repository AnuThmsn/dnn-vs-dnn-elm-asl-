"""
generate_calibration_dataset.py
Generates the separate, independent residual-calibration dataset for V2:
- 100,000 observations
- Same Buxton kinetic model
- Same four PLDs: [1.525, 2.025, 2.525, 3.025]
- Same physical constants
- Same CBF [0, 100] and ATT [0.5, 3.0] ranges
- Dual-channel quadrature Rician magnitude noise
- SNR distribution uniformly spanning 10 to 20 dB
- Independent CBF-ATT sampling
- Reproducible random seed: 31415
"""

import os
import sys
import numpy as np

# Add repository root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)

from experiments.physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    compute_physical_signals, add_rician_noise
)

def generate_calibration_dataset(
    output_path=None,
    n_samples=100000,
    snr_min=10.0,
    snr_max=20.0,
    seed=31415
):
    if output_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "data"))
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, "residual_calibration_100k.npz")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating independent residual-calibration dataset:")
    print(f"  Samples : {n_samples:,}")
    print(f"  SNR     : Uniform in [{snr_min:.1f}, {snr_max:.1f}] dB")
    print(f"  PLDs    : {PLDS.tolist()} s")
    print(f"  CBF     : [{CBF_MIN}, {CBF_MAX}] mL/100g/min")
    print(f"  ATT     : [{ATT_MIN}, {ATT_MAX}] s")
    print(f"  Seed    : {seed}")

    rng = np.random.default_rng(seed)

    # 1. Independent parameter sampling
    cbf = rng.uniform(CBF_MIN, CBF_MAX, n_samples).astype(np.float64)
    att = rng.uniform(ATT_MIN, ATT_MAX, n_samples).astype(np.float64)
    snr = rng.uniform(snr_min, snr_max, n_samples).astype(np.float64)

    # 2. Buxton physical signal generation
    clean_sig = compute_physical_signals(PLDS, TAU, cbf, att)

    # 3. Dual-channel quadrature Rician noise
    noisy_x = add_rician_noise(clean_sig, snr, rng)

    y = np.stack([cbf, att], axis=1).astype(np.float32)
    noisy_x = noisy_x.astype(np.float32)

    # 4. Save compressed dataset
    np.savez_compressed(
        output_path,
        X_cal=noisy_x,
        Y_cal=y,
        snr_cal=snr,
        clean_cal=clean_sig.astype(np.float32),
        plds=PLDS,
        seed=seed
    )

    print(f"Successfully generated and saved calibration dataset to:")
    print(f"  {output_path}")
    print(f"  X_cal shape: {noisy_x.shape}, Y_cal shape: {y.shape}, snr_cal shape: {snr.shape}")
    return noisy_x, y, snr, output_path

if __name__ == "__main__":
    generate_calibration_dataset()

