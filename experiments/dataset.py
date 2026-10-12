"""
dataset.py
Ultra-low-RAM, disk-cached data generation pipeline reproducing Experiment H exactly.
Stores all cache files on Drive D without touching Drive C.
Peak RAM consumption is capped under 100 MB.
"""

import os
import gc
import json
import numpy as np

# Enforce Drive D environment first
import memory_config
from physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SNR_VALUES, compute_physical_signals, add_rician_noise
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def generate_and_cache_split(n_samples, seed, split_name, chunk_size=200_000):
    """
    Generate dataset in low-RAM chunks and save directly to disk on Drive D.
    Returns file paths for memory-mapped loading.
    """
    x_path = os.path.join(CACHE_DIR, f"X_{split_name}.npy")
    y_path = os.path.join(CACHE_DIR, f"Y_{split_name}.npy")
    snr_path = os.path.join(CACHE_DIR, f"snr_{split_name}.npy")

    # If already cached and complete, return paths
    if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(snr_path):
        print(f"Loading existing cache for {split_name} from {CACHE_DIR}...")
        return x_path, y_path, snr_path

    print(f"Generating {split_name} (N={n_samples:,}, seed={seed}) in chunks of {chunk_size:,}...")
    rng = np.random.default_rng(seed)
    snr_probs = [1.0 / len(SNR_VALUES)] * len(SNR_VALUES)

    # Pre-allocate numpy files using memory mapping on Drive D
    x_mmap = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float32, shape=(n_samples, 4))
    y_mmap = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.float32, shape=(n_samples, 2))
    snr_mmap = np.lib.format.open_memmap(snr_path, mode='w+', dtype=np.float64, shape=(n_samples,))

    n_chunks = int(np.ceil(n_samples / chunk_size))
    for c_idx in range(n_chunks):
        start = c_idx * chunk_size
        end = min(start + chunk_size, n_samples)
        c_len = end - start

        cbf_c = rng.uniform(CBF_MIN, CBF_MAX, c_len).astype(np.float64)
        att_c = rng.uniform(ATT_MIN, ATT_MAX, c_len).astype(np.float64)
        snr_c = rng.choice(SNR_VALUES, size=c_len, p=snr_probs)

        clean_sig_c = compute_physical_signals(PLDS, TAU, cbf_c, att_c)
        X_c = add_rician_noise(clean_sig_c, snr_c, rng)

        x_mmap[start:end] = X_c
        y_mmap[start:end, 0] = cbf_c
        y_mmap[start:end, 1] = att_c
        snr_mmap[start:end] = snr_c

        # Flush chunk to Drive D disk and free RAM
        x_mmap.flush()
        y_mmap.flush()
        snr_mmap.flush()
        del cbf_c, att_c, snr_c, clean_sig_c, X_c
        gc.collect()

    del x_mmap, y_mmap, snr_mmap
    gc.collect()
    print(f"Finished caching {split_name} to Drive D.")
    return x_path, y_path, snr_path


def generate_clean_grid():
    """
    Generate 2,392 clean physiological grid points (46 CBF x 52 ATT)
    matching Experiment G / H exactly.
    """
    cbf_vals = np.linspace(10.0, 100.0, 46)
    att_vals = np.linspace(0.5, 3.0, 52)
    cbf_grid, att_grid = np.meshgrid(cbf_vals, att_vals, indexing="ij")
    cbf_flat = cbf_grid.ravel()
    att_flat = att_grid.ravel()

    clean_sig = compute_physical_signals(PLDS, TAU, cbf_flat, att_flat)
    X_clean = (2.0 * clean_sig).astype(np.float32)
    Y_clean = np.stack([cbf_flat, att_flat], axis=1).astype(np.float32)
    return X_clean, Y_clean, clean_sig, cbf_flat, att_flat


def prepare_and_validate_all_datasets():
    """Prepares and validates all splits with low RAM usage."""
    memory_config.print_memory_report("Before Dataset Preparation")

    # 1. Generate & Cache on Drive D
    x_tr_path, y_tr_path, snr_tr_path = generate_and_cache_split(1_000_000, seed=42, split_name="train")
    x_val_path, y_val_path, snr_val_path = generate_and_cache_split(100_000, seed=99, split_name="val")
    x_test_path, y_test_path, snr_test_path = generate_and_cache_split(100_000, seed=7, split_name="test")

    # 2. Memory-map arrays for fast zero-RAM validation
    X_tr = np.load(x_tr_path, mmap_mode='r')
    Y_tr = np.load(y_tr_path, mmap_mode='r')
    snr_tr = np.load(snr_tr_path, mmap_mode='r')

    X_val = np.load(x_val_path, mmap_mode='r')
    Y_val = np.load(y_val_path, mmap_mode='r')

    X_test = np.load(x_test_path, mmap_mode='r')
    Y_test = np.load(y_test_path, mmap_mode='r')

    print("\n--- Shape and Bound Validation ---")
    assert X_tr.shape == (1_000_000, 4) and Y_tr.shape == (1_000_000, 2)
    assert X_val.shape == (100_000, 4) and Y_val.shape == (100_000, 2)
    assert X_test.shape == (100_000, 4) and Y_test.shape == (100_000, 2)
    print(f"Shapes verified: Train {X_tr.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # Check bounds (sampling 10k items to avoid heavy RAM load)
    assert 0.0 <= Y_tr[:10000, 0].min() and Y_tr[:10000, 0].max() <= 100.0
    assert 0.5 <= Y_tr[:10000, 1].min() and Y_tr[:10000, 1].max() <= 3.0
    print("Bound verification: PASSED")

    # 3. Compute Normalization Parameters across Train (chunked for minimal RAM)
    print("\nComputing normalization statistics from Train (chunked)...")
    sum_x = np.zeros(4, dtype=np.float64)
    sum_sq_x = np.zeros(4, dtype=np.float64)
    sum_cbf = 0.0
    sum_sq_cbf = 0.0
    sum_att = 0.0
    sum_sq_att = 0.0

    chunk = 100_000
    N = 1_000_000
    for i in range(0, N, chunk):
        x_c = X_tr[i:i+chunk]
        y_c = Y_tr[i:i+chunk]
        sum_x += x_c.sum(axis=0)
        sum_sq_x += (x_c.astype(np.float64)**2).sum(axis=0)
        sum_cbf += y_c[:, 0].sum()
        sum_sq_cbf += (y_c[:, 0].astype(np.float64)**2).sum()
        sum_att += y_c[:, 1].sum()
        sum_sq_att += (y_c[:, 1].astype(np.float64)**2).sum()

    X_mean = (sum_x / N).astype(np.float32).reshape(1, 4)
    X_std = np.sqrt((sum_sq_x / N) - (X_mean.squeeze()**2)).astype(np.float32).reshape(1, 4)
    cbf_mean = float(sum_cbf / N)
    cbf_std = float(np.sqrt((sum_sq_cbf / N) - (cbf_mean**2)))
    att_mean = float(sum_att / N)
    att_std = float(np.sqrt((sum_sq_att / N) - (att_mean**2)))

    print(f"X_mean: {X_mean.squeeze()}")
    print(f"X_std : {X_std.squeeze()}")
    print(f"CBF mean/std: {cbf_mean:.4f} / {cbf_std:.4f}")
    print(f"ATT mean/std: {att_mean:.4f} / {att_std:.4f}")

    # Clean Grid
    X_clean, Y_clean, clean_sig, cbf_flat, att_flat = generate_clean_grid()
    clean_cache_path = os.path.join(CACHE_DIR, "clean_grid.npz")
    np.savez(clean_cache_path, X_clean=X_clean, Y_clean=Y_clean,
             clean_sig=clean_sig, cbf_flat=cbf_flat, att_flat=att_flat)
    print(f"Clean grid cached to {clean_cache_path}")

    # Save normalization stats to cache
    norm_path = os.path.join(CACHE_DIR, "normalization_stats.npz")
    np.savez(norm_path, X_mean=X_mean, X_std=X_std,
             cbf_mean=cbf_mean, cbf_std=cbf_std, att_mean=att_mean, att_std=att_std)
    print(f"Normalization stats saved to {norm_path}")

    memory_config.print_memory_report("After Dataset Preparation")
    return {
        "x_tr_path": x_tr_path, "y_tr_path": y_tr_path,
        "x_val_path": x_val_path, "y_val_path": y_val_path,
        "x_test_path": x_test_path, "y_test_path": y_test_path,
        "X_mean": X_mean, "X_std": X_std,
        "cbf_mean": cbf_mean, "cbf_std": cbf_std,
        "att_mean": att_mean, "att_std": att_std
    }


if __name__ == "__main__":
    prepare_and_validate_all_datasets()
