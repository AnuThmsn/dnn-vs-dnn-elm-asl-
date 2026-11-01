"""
generate_independent_test.py
Generates a completely independent 30,000-sample test cohort:
- 10,000 observations at SNR 10
- 10,000 observations at SNR 15
- 10,000 observations at SNR 20
Using independent ground-truth CBF and ATT parameters sampled from U[0, 100] and U[0.5, 3.0].
Dual-channel quadrature Rician noise is applied with exact Buxton reference scaling (SIG_REF_SC).
Held-out random seed: 2026.
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

def generate_independent_test_set(output_dir=None, seed=2026, samples_per_snr=10000, snr_levels=(10.0, 15.0, 20.0)):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    
    X_list = []
    Y_list = []
    snr_list = []

    print(f"Generating independent test cohort (Seed {seed}):")
    for snr in snr_levels:
        cbf = rng.uniform(CBF_MIN, CBF_MAX, samples_per_snr).astype(np.float64)
        att = rng.uniform(ATT_MIN, ATT_MAX, samples_per_snr).astype(np.float64)
        
        clean_sig = compute_physical_signals(PLDS, TAU, cbf, att)
        noisy_x = add_rician_noise(clean_sig, snr, rng)
        
        y = np.stack([cbf, att], axis=1).astype(np.float32)
        snr_arr = np.full(samples_per_snr, snr, dtype=np.float64)
        
        X_list.append(noisy_x)
        Y_list.append(y)
        snr_list.append(snr_arr)
        print(f"  SNR {snr:4.1f}: Generated {samples_per_snr:,} unique observations.")

    X_test_all = np.concatenate(X_list, axis=0).astype(np.float32)
    Y_test_all = np.concatenate(Y_list, axis=0).astype(np.float32)
    snr_test_all = np.concatenate(snr_list, axis=0).astype(np.float64)

    test_save_path = os.path.join(output_dir, "independent_test_30k.npz")
    np.savez_compressed(
        test_save_path,
        X_test=X_test_all,
        Y_test=Y_test_all,
        snr_test=snr_test_all,
        plds=PLDS,
        seed=seed
    )
    print(f"Saved 30,000 independent test samples to {test_save_path}")
    print(f"  X shape: {X_test_all.shape}, Y shape: {Y_test_all.shape}")
    return X_test_all, Y_test_all, snr_test_all, test_save_path

if __name__ == "__main__":
    generate_independent_test_set()

