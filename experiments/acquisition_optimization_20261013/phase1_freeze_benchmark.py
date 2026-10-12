"""
phase1_freeze_benchmark.py
Phase 1: Freeze and record baseline benchmark hashes and configurations.
Ensures zero alteration to existing baseline artifacts and records
the authoritative fixed reference for the acquisition redesign campaign.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
ELM_DIR = os.path.join(EXP_DIR, "elm_models")
OUT_DIR = os.path.join(EXP_DIR, "acquisition_optimization_20261013")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
LOGS_DIR = os.path.join(OUT_DIR, "logs")

for d in [OUT_DIR, TABLES_DIR, PLOTS_DIR, CKPT_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.append(EXP_DIR)
import physics

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("PHASE 1: RECORDING BASELINE BENCHMARK AND PRESERVATION AUDIT")
    print("=" * 80)

    # 1. Baseline checkpoint hashes
    att_ckpt = os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth")
    cbf_ckpt = os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth")
    norm_path = os.path.join(BASE_DIR, "normalization_stats.npz")
    preds_path = os.path.join(ELM_DIR, "all_test_predictions.npz")

    files_to_hash = {
        "ATTnet_G_baselineA.pth": att_ckpt,
        "CBFnet_G_baselineA.pth": cbf_ckpt,
        "normalization_stats.npz": norm_path,
        "all_test_predictions.npz": preds_path
    }

    hashes = {}
    for name, path in files_to_hash.items():
        if os.path.exists(path):
            h = sha256_file(path)
            hashes[name] = {"path": path, "sha256": h, "size_bytes": os.path.getsize(path)}
            print(f"  {name}: {h}")
        else:
            print(f"  WARNING: {name} not found at {path}")

    # 2. Frozen Baseline Acquisition Protocol
    baseline_protocol = {
        "plds_seconds": physics.PLDS.tolist(),
        "tau_seconds": physics.TAU,
        "T1t_seconds": physics.T1T,
        "T1a_seconds": physics.T1A,
        "alpha": physics.ALPHA,
        "beta": physics.BETA,
        "lambda_ml_g": physics.LAMBDA,
        "scale": physics.SCALE,
        "sig_ref_sc": physics.SIG_REF_SC,
        "cbf_min": physics.CBF_MIN,
        "cbf_max": physics.CBF_MAX,
        "att_min": physics.ATT_MIN,
        "att_max": physics.ATT_MAX,
        "snr_values": [float(s) if not np.isinf(s) else "inf" for s in physics.SNR_VALUES]
    }

    # 3. Authoritative Baseline Performance (from final evaluation on N=100,000)
    final_test_csv = os.path.join(EXP_DIR, "significant_improvement_20261012", "tables", "final_candidates_test_comparison.csv")
    if os.path.exists(final_test_csv):
        df_final = pd.read_csv(final_test_csv)
        base_row = df_final[df_final["Candidate_Model"] == "Baseline A (DNN-G)"].to_dict(orient="records")[0]
        ens_row = df_final[df_final["Candidate_Model"] == "Ensemble Feature ELM (Baseline Latents)"].to_dict(orient="records")[0]
    else:
        base_row = {}
        ens_row = {}

    benchmark_record = {
        "campaign": "Acquisition Optimization 20261013",
        "reference_baseline": "Baseline A (DNN-G) and Ensemble Feature ELM",
        "file_hashes": hashes,
        "baseline_protocol": baseline_protocol,
        "baseline_A_test_metrics": base_row,
        "ensemble_feature_elm_test_metrics": ens_row
    }

    out_json = os.path.join(TABLES_DIR, "phase1_benchmark_frozen.json")
    with open(out_json, "w") as f:
        json.dump(benchmark_record, f, indent=2)

    print(f"\nFrozen benchmark metadata successfully saved to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()

