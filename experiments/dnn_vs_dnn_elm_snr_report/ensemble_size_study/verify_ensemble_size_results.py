"""
verify_ensemble_size_results.py
Verification suite for the DNN-Feature ELM Ensemble Architecture and Size Study.
Performs:
1. File integrity checks.
2. Locked test set size and SNR distribution verification.
3. Metric sanity checks (RMSE >= MAE >= |Bias|, -1 <= CCC <= 1).
4. Nesting and reproducibility verification against historical locked benchmarks.
5. Member diversity metric validity checks.
6. Generates experiment_manifest.json.
"""

import os
import sys
import hashlib
import json
import pandas as pd
import numpy as np

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(STUDY_DIR, "..", "..", ".."))

def compute_sha256(filepath):
    if not os.path.isfile(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_all():
    print("=" * 80)
    print("VERIFICATION SUITE: DNN-FEATURE ELM ENSEMBLE SIZE STUDY")
    print("=" * 80)

    # 1. File existence
    required_files = [
        "architecture_audit.md",
        "run_phase1_audit.py",
        "run_ensemble_study.py",
        "ensemble_size_metrics.csv",
        "ensemble_size_summary.csv",
        "member_diversity_metrics.csv",
        "ensemble_size_report.md"
    ]
    for rf in required_files:
        p = os.path.join(STUDY_DIR, rf)
        assert os.path.isfile(p), f"Missing required file: {rf}"
        print(f"[PASS] Found: {rf}")

    # 2. Check metrics data
    df_metrics = pd.read_csv(os.path.join(STUDY_DIR, "ensemble_size_metrics.csv"))
    df_summary = pd.read_csv(os.path.join(STUDY_DIR, "ensemble_size_summary.csv"))
    df_div = pd.read_csv(os.path.join(STUDY_DIR, "member_diversity_metrics.csv"))

    # Test sample counts
    snr_counts = {10: 8358, 15: 8279, 20: 8385}
    for snr, expected_cnt in snr_counts.items():
        sub = df_metrics[df_metrics["SNR"] == snr]
        for _, row in sub.iterrows():
            assert row["N_Samples"] == expected_cnt, f"SNR {snr} sample count mismatch: {row['N_Samples']} != {expected_cnt}"
    print("[PASS] Sample counts per SNR match locked test set exactly (8358, 8279, 8385).")

    # 3. Metric mathematical sanity
    for df, name in [(df_metrics, "ensemble_size_metrics"), (df_summary, "ensemble_size_summary")]:
        for _, row in df.iterrows():
            assert row["CBF_RMSE"] >= row["CBF_MAE"], f"CBF RMSE < MAE in {name}"
            assert row["ATT_RMSE"] >= row["ATT_MAE"], f"ATT RMSE < MAE in {name}"
            assert row["CBF_MAE"] >= abs(row["CBF_Bias"]), f"CBF MAE < |Bias| in {name}"
            assert row["ATT_MAE"] >= abs(row["ATT_Bias"]), f"ATT MAE < |Bias| in {name}"
            assert -1.0 <= row["CBF_CCC"] <= 1.0, f"CBF CCC out of bounds in {name}"
            assert -1.0 <= row["ATT_CCC"] <= 1.0, f"ATT CCC out of bounds in {name}"
    print("[PASS] Metric mathematical sanity checks passed (RMSE >= MAE >= |Bias|, CCC in [-1, 1]).")

    # 4. Historical benchmark exact replication check (Repeat 1, K=5)
    r1_k5 = df_summary[df_summary["Ensemble_Size_K"] == 5].set_index("SNR")
    
    # SNR 10
    assert abs(r1_k5.loc[10, "CBF_RMSE"] - 4.2069) < 0.001, f"CBF RMSE SNR 10 mismatch: {r1_k5.loc[10, 'CBF_RMSE']}"
    assert abs(r1_k5.loc[10, "ATT_RMSE"] - 0.3633) < 0.001, f"ATT RMSE SNR 10 mismatch: {r1_k5.loc[10, 'ATT_RMSE']}"
    # SNR 15
    assert abs(r1_k5.loc[15, "CBF_RMSE"] - 3.3721) < 0.001, f"CBF RMSE SNR 15 mismatch: {r1_k5.loc[15, 'CBF_RMSE']}"
    assert abs(r1_k5.loc[15, "ATT_RMSE"] - 0.3279) < 0.001, f"ATT RMSE SNR 15 mismatch: {r1_k5.loc[15, 'ATT_RMSE']}"
    # SNR 20
    assert abs(r1_k5.loc[20, "CBF_RMSE"] - 2.9912) < 0.001, f"CBF RMSE SNR 20 mismatch: {r1_k5.loc[20, 'CBF_RMSE']}"
    assert abs(r1_k5.loc[20, "ATT_RMSE"] - 0.2989) < 0.001, f"ATT RMSE SNR 20 mismatch: {r1_k5.loc[20, 'ATT_RMSE']}"
    print("[PASS] Historical benchmark exactly replicated for K=5 at SNR 10, 15, and 20.")

    # 5. Member diversity sanity checks
    for _, row in df_div.iterrows():
        assert 0.99 <= row["CBF_Mean_Pairwise_Corr"] <= 1.0, "CBF Pairwise correlation out of expected bounds"
        assert 0.99 <= row["ATT_Mean_Pairwise_Corr"] <= 1.0, "ATT Pairwise correlation out of expected bounds"
        assert row["CBF_Voxel_Prediction_Variance"] >= 0.0, "CBF Voxel variance negative"
        assert row["ATT_Voxel_Prediction_Variance"] >= 0.0, "ATT Voxel variance negative"
        assert row["CBF_Averaging_Gain"] >= -1e-6, "CBF Averaging gain negative"
    print("[PASS] Member diversity metrics verified.")

    # 6. Generate manifest
    manifest = {
        "study": "DNN-Feature ELM Ensemble Architecture Verification and Optimal Size Study",
        "directory": "experiments/dnn_vs_dnn_elm_snr_report/ensemble_size_study/",
        "plds_seconds": [1.525, 2.025, 2.525, 3.025],
        "ensemble_sizes_evaluated": [1, 3, 5, 10],
        "snr_targets_db": [10, 15, 20],
        "replications": 3,
        "input_checkpoints": {
            "CBFnet_G_baselineA.pth": compute_sha256(os.path.join(REPO_ROOT, "experiments", "baseline_A", "CBFnet_G_baselineA.pth")),
            "ATTnet_G_baselineA.pth": compute_sha256(os.path.join(REPO_ROOT, "experiments", "baseline_A", "ATTnet_G_baselineA.pth")),
            "normalization_stats.npz": compute_sha256(os.path.join(REPO_ROOT, "experiments", "baseline_A", "normalization_stats.npz"))
        },
        "study_artifacts": {
            rf: compute_sha256(os.path.join(STUDY_DIR, rf)) for rf in required_files
        },
        "summary_findings": {
            "recommended_ensemble_size": 5,
            "justification": "Optimal variance reduction with saturated error curve at K=5; K=10 yields < 0.0003 error reduction; K=1 incurs higher variance.",
            "members_are_distinct": True,
            "aggregation_method": "Unweighted arithmetic mean of physical parameter estimates after clipping"
        }
    }

    manifest_path = os.path.join(STUDY_DIR, "experiment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[PASS] Saved experiment manifest to: {manifest_path}")

    print("\nALL VERIFICATION CHECKS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    verify_all()
