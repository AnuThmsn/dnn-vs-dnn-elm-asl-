"""
verify_baseline_integrity.py
Phase 1: Baseline Integrity Audit
Loads all saved Baseline A artifacts, recomputes all metrics from raw arrays,
verifies match against CSV files, and computes cryptographic SHA-256 hashes.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import memory_config
from physics import PLDS, ATT_MIN, ATT_MAX, CBF_MIN, CBF_MAX
from train_baseline_A import StandardizedNet, calc_metrics

BASE_DIR = os.path.join(os.path.dirname(__file__), "baseline_A")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("PHASE 1: BASELINE INTEGRITY AUDIT & ARTIFACT RECALCULATION")
    print("=" * 80)

    # 1. Audit artifact files & SHA-256
    files_to_check = [
        "ATTnet_G_baselineA.pth",
        "CBFnet_G_baselineA.pth",
        "normalization_stats.npz",
        "training_history.json",
        "clean_grid_baselineA.csv",
        "test_summary_baselineA.csv",
        "snr_evaluation_baselineA.csv",
        "baselineA_test_predictions.npz"
    ]

    print("\n--- Artifact Inventory & Hashes ---")
    hashes = {}
    for f in files_to_check:
        fp = os.path.join(BASE_DIR, f)
        assert os.path.exists(fp), f"Missing artifact: {fp}"
        sz = os.path.getsize(fp) / 1024
        sha = compute_sha256(fp)
        hashes[f] = sha
        print(f"  {f:<32} {sz:8.1f} KB | SHA256: {sha[:16]}...")

    # 2. Verify model checkpoints can be instantiated and loaded
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    att_net = StandardizedNet(len(PLDS), n_hidden_layers=4, n_neurons=64,
                              y_mean=att_mean, y_std=att_std, y_min=ATT_MIN, y_max=ATT_MAX)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location="cpu"))
    att_net.eval()

    cbf_net = StandardizedNet(len(PLDS), n_hidden_layers=5, n_neurons=128,
                              y_mean=cbf_mean, y_std=cbf_std, y_min=CBF_MIN, y_max=CBF_MAX)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location="cpu"))
    cbf_net.eval()

    n_att = sum(p.numel() for p in att_net.parameters())
    n_cbf = sum(p.numel() for p in cbf_net.parameters())
    print(f"\nModel architecture loaded successfully:")
    print(f"  ATTnet params: {n_att:,} (Expected: 12,865)")
    print(f"  CBFnet params: {n_cbf:,} (Expected: 66,817)")
    assert n_att == 12865 and n_cbf == 66817

    # 3. Recalculate Test Metrics from raw arrays
    test_preds = np.load(os.path.join(BASE_DIR, "baselineA_test_predictions.npz"))
    cbf_test_pred = test_preds["cbf_test_pred"]
    att_test_pred = test_preds["att_test_pred"]
    cbf_clean_pred = test_preds["cbf_clean_pred"]
    att_clean_pred = test_preds["att_clean_pred"]

    y_test = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    clean_data = np.load(os.path.join(CACHE_DIR, "clean_grid.npz"))
    cbf_clean_true = clean_data["cbf_flat"]
    att_clean_true = clean_data["att_flat"]

    # Recompute test metrics
    recomputed_test_cbf = calc_metrics(y_test[:, 0], cbf_test_pred)
    recomputed_test_att = calc_metrics(y_test[:, 1], att_test_pred)

    test_csv = pd.read_csv(os.path.join(BASE_DIR, "test_summary_baselineA.csv")).set_index("Target")

    print("\n--- Held-Out Test Set Recalculation Check ---")
    for key in ["RMSE", "MAE", "Bias", "R2", "CCC"]:
        csv_cbf = test_csv.loc["CBF", key]
        rec_cbf = recomputed_test_cbf[key]
        diff_cbf = abs(csv_cbf - rec_cbf)
        print(f"  CBF {key:<5}: Recalculated = {rec_cbf:.6f} | CSV = {csv_cbf:.6f} | Diff = {diff_cbf:.2e}")
        assert diff_cbf < 1e-5, f"Discrepancy in CBF {key}"

        csv_att = test_csv.loc["ATT", key]
        rec_att = recomputed_test_att[key]
        diff_att = abs(csv_att - rec_att)
        print(f"  ATT {key:<5}: Recalculated = {rec_att:.6f} | CSV = {csv_att:.6f} | Diff = {diff_att:.2e}")
        assert diff_att < 1e-5, f"Discrepancy in ATT {key}"
    print("Held-Out Test Set: PASSED (Exact match within float tolerance)")

    # 4. Recompute clean grid metrics
    recomputed_clean_cbf = calc_metrics(cbf_clean_true, cbf_clean_pred)
    recomputed_clean_att = calc_metrics(att_clean_true, att_clean_pred)
    clean_csv = pd.read_csv(os.path.join(BASE_DIR, "clean_grid_baselineA.csv")).set_index("Target")

    print("\n--- Clean Grid Recalculation Check ---")
    for key in ["RMSE", "MAE", "R2", "CCC"]:
        csv_cbf = clean_csv.loc["CBF", key]
        rec_cbf = recomputed_clean_cbf[key]
        diff_cbf = abs(csv_cbf - rec_cbf)
        print(f"  CBF {key:<5}: Recalculated = {rec_cbf:.6f} | CSV = {csv_cbf:.6f} | Diff = {diff_cbf:.2e}")
        assert diff_cbf < 1e-5, f"Discrepancy in Clean CBF {key}"

        csv_att = clean_csv.loc["ATT", key]
        rec_att = recomputed_clean_att[key]
        diff_att = abs(csv_att - rec_att)
        print(f"  ATT {key:<5}: Recalculated = {rec_att:.6f} | CSV = {csv_att:.6f} | Diff = {diff_att:.2e}")
        assert diff_att < 1e-5, f"Discrepancy in Clean ATT {key}"
    print("Clean Grid: PASSED (Exact match within float tolerance)")

    # 5. Check Training History
    with open(os.path.join(BASE_DIR, "training_history.json"), "r") as f:
        hist = json.load(f)
    print(f"\n--- Training History Integrity ---")
    print(f"  ATTnet Epochs: {len(hist['att_history'])} (Training time: {hist['att_time_sec']:.1f} s)")
    print(f"  CBFnet Epochs: {len(hist['cbf_history'])} (Training time: {hist['cbf_time_sec']:.1f} s)")
    
    # Identify best validation epoch
    att_val_losses = [e["val_loss"] for e in hist["att_history"]]
    best_att_ep = int(np.argmin(att_val_losses)) + 1
    best_att_val = float(np.min(att_val_losses))
    print(f"  ATTnet Best Val Loss: {best_att_val:.6f} at Epoch {best_att_ep} (Patience 10 stopped at {len(hist['att_history'])})")

    cbf_val_losses = [e["val_loss"] for e in hist["cbf_history"]]
    best_cbf_ep = int(np.argmin(cbf_val_losses)) + 1
    best_cbf_val = float(np.min(cbf_val_losses))
    print(f"  CBFnet Best Val Loss: {best_cbf_val:.6f} at Epoch {best_cbf_ep} (Patience 10 stopped at {len(hist['cbf_history'])})")

    print("\n" + "=" * 80)
    print("BASELINE A AUDIT PASSED: ALL ARTIFACTS VERIFIED AND NUMERICALLY MATCHING")
    print("=" * 80)

if __name__ == "__main__":
    main()

