"""
phase3_pld_search.py
Phase 3: Search for Optimal PLD Acquisition Configurations.
Systematic grid search over physically valid, scanner-constrained PLD combinations.
Accounts for:
  - Scanner timing: PLD1 >= 0.40s, spacing >= 0.35s, PLD4 <= 3.025s.
  - Acquisition budget: Total PLD sum <= baseline sum (9.10s) for equal or lower scan time.
  - Cramér-Rao Lower Bounds (CRLB) for CBF and ATT over physiological distribution.
  - Condition number log10 kappa(J) and D-optimality log(det(F)).
  - Targeted early-ATT ([0.5, 1.0)s) and full-range performance.
  - Secondary study: 3-PLD configurations (25% scan time savings).
Saves tables/pld_configuration_search.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
OUT_DIR = os.path.join(EXP_DIR, "acquisition_optimization_20261013")
TABLES_DIR = os.path.join(OUT_DIR, "tables")

sys.path.append(EXP_DIR)
sys.path.append(OUT_DIR)
import physics
from phase2_identifiability import compute_analytical_jacobian, evaluate_jacobian_properties

def evaluate_pld_candidate(plds, eval_grid, sigma_noise):
    """
    Evaluates a candidate PLD vector across a grid of (CBF, ATT) points.
    Returns composite summary metrics.
    """
    crlb_att_list = []
    crlb_cbf_list = []
    log_cond_list = []
    log_det_list = []
    collin_list = []

    crlb_att_early_list = []
    collin_early_list = []

    for cbf_val, att_val in eval_grid:
        J = compute_analytical_jacobian(plds, physics.TAU, cbf_val, att_val)
        props = evaluate_jacobian_properties(J, sigma_noise=sigma_noise)

        # Cap singular/infinite CRLBs at 10.0 for robust scoring
        c_att = min(props["crlb_att"], 10.0) if not np.isinf(props["crlb_att"]) else 10.0
        c_cbf = min(props["crlb_cbf"], 100.0) if not np.isinf(props["crlb_cbf"]) else 100.0
        log_k = np.log10(min(props["cond_J"], 1e16))
        det_f = max(props["det_F"], 1e-12)
        collin = props["cos_collinearity"]

        crlb_att_list.append(c_att)
        crlb_cbf_list.append(c_cbf)
        log_cond_list.append(log_k)
        log_det_list.append(np.log(det_f))
        collin_list.append(collin)

        if att_val < 1.0:
            crlb_att_early_list.append(c_att)
            collin_early_list.append(collin)

    mean_crlb_att = float(np.mean(crlb_att_list))
    mean_crlb_cbf = float(np.mean(crlb_cbf_list))
    mean_log_cond = float(np.mean(log_cond_list))
    mean_log_det = float(np.mean(log_det_list))
    mean_collin = float(np.mean(collin_list))

    early_crlb_att = float(np.mean(crlb_att_early_list))
    early_collin = float(np.mean(collin_early_list))

    # Composite Score: lower is better
    # Penalizes high ATT uncertainty (especially early), high CBF uncertainty, and ill-conditioning
    score = (
        early_crlb_att * 2.0 +
        mean_crlb_att * 1.0 +
        mean_crlb_cbf * 0.05 +
        mean_log_cond * 0.1
    )

    return {
        "Score": score,
        "Early_ATT_CRLB": early_crlb_att,
        "Early_Collinearity": early_collin,
        "Mean_ATT_CRLB": mean_crlb_att,
        "Mean_CBF_CRLB": mean_crlb_cbf,
        "Mean_Log10_Cond": mean_log_cond,
        "Mean_Log_Det_FIM": mean_log_det,
        "Mean_Collinearity": mean_collin
    }

def main():
    print("=" * 80)
    print("PHASE 3: SEARCH FOR OPTIMAL PLD CONFIGURATIONS")
    print("=" * 80)

    # 1. Construct Evaluation Grid over physiological parameter space
    cbf_pts = [20.0, 40.0, 60.0, 80.0]
    att_pts = np.linspace(0.55, 2.95, 25) # 25 ATT values across full range
    eval_grid = [(c, a) for c in cbf_pts for a in att_pts] # 100 points
    sigma_ref_20 = physics.SIG_REF_SC / 20.0 # SNR = 20

    baseline_plds = np.array([1.525, 2.025, 2.525, 3.025], dtype=np.float64)
    baseline_sum = float(np.sum(baseline_plds)) # 9.10 s

    # 2. Build Candidate Search Space
    # Search candidates with PLD1 from 0.40 to 0.80s, PLD4 up to 3.025s
    pld1_candidates = [0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.525]
    pld2_candidates = [0.90, 1.10, 1.20, 1.30, 1.50, 1.70, 2.025]
    pld3_candidates = [1.50, 1.80, 2.00, 2.20, 2.40, 2.525]
    pld4_candidates = [2.20, 2.50, 2.70, 2.80, 3.00, 3.025]

    valid_candidates_4 = []

    for w1 in pld1_candidates:
        for w2 in pld2_candidates:
            if w2 < w1 + 0.35:
                continue
            for w3 in pld3_candidates:
                if w3 < w2 + 0.35:
                    continue
                for w4 in pld4_candidates:
                    if w4 < w3 + 0.35:
                        continue
                    tot_sum = w1 + w2 + w3 + w4
                    if tot_sum <= baseline_sum + 0.10: # Scanner budget constraint
                        valid_candidates_4.append(np.array([w1, w2, w3, w4], dtype=np.float64))

    print(f"Total valid 4-PLD candidate configurations generated: {len(valid_candidates_4)}")

    # Secondary study: 3-PLD configurations
    valid_candidates_3 = []
    pld1_3 = [0.40, 0.50, 0.60, 0.70]
    pld2_3 = [1.20, 1.40, 1.60, 1.80]
    pld3_3 = [2.40, 2.60, 2.80, 3.00]
    for w1 in pld1_3:
        for w2 in pld2_3:
            for w3 in pld3_3:
                valid_candidates_3.append(np.array([w1, w2, w3], dtype=np.float64))
    print(f"Total valid 3-PLD candidate configurations generated: {len(valid_candidates_3)}")

    # 3. Evaluate all 4-PLD candidates
    rows = []
    for cand in valid_candidates_4:
        is_baseline = np.allclose(cand, baseline_plds)
        res = evaluate_pld_candidate(cand, eval_grid, sigma_noise=sigma_ref_20)
        pld_str = "[" + ", ".join([f"{x:.3f}" if abs(x - round(x, 1)) > 0.01 else f"{x:.2f}" for x in cand]) + "]"
        rows.append({
            "Configuration": pld_str,
            "N_PLD": 4,
            "PLD_1": cand[0], "PLD_2": cand[1], "PLD_3": cand[2], "PLD_4": cand[3],
            "Total_PLD_Sum": float(np.sum(cand)),
            "Is_Baseline": is_baseline,
            **res
        })

    # Evaluate 3-PLD candidates
    for cand in valid_candidates_3:
        res = evaluate_pld_candidate(cand, eval_grid, sigma_noise=sigma_ref_20)
        pld_str = "[" + ", ".join([f"{x:.2f}" for x in cand]) + "]"
        rows.append({
            "Configuration": pld_str,
            "N_PLD": 3,
            "PLD_1": cand[0], "PLD_2": cand[1], "PLD_3": cand[2], "PLD_4": np.nan,
            "Total_PLD_Sum": float(np.sum(cand)),
            "Is_Baseline": False,
            **res
        })

    df_search = pd.DataFrame(rows)
    df_search.sort_values(by=["N_PLD", "Score"], ascending=[False, True], inplace=True)

    csv_out = os.path.join(TABLES_DIR, "pld_configuration_search.csv")
    df_search.to_csv(csv_out, index=False)
    print(f"\nSaved search results to: {csv_out}")

    # Display Top 4-PLD Configurations vs Baseline Reference
    df_4 = df_search[df_search["N_PLD"] == 4].copy()
    print("\n--- Top 5 Optimized 4-PLD Configurations ---")
    print(df_4.head(5)[["Configuration", "Total_PLD_Sum", "Score", "Early_ATT_CRLB", "Mean_ATT_CRLB", "Mean_CBF_CRLB", "Mean_Log10_Cond"]].to_string(index=False))

    print("\n--- Baseline Reference (Rank in search) ---")
    df_base = df_4[df_4["Is_Baseline"] == True]
    print(df_base[["Configuration", "Total_PLD_Sum", "Score", "Early_ATT_CRLB", "Mean_ATT_CRLB", "Mean_CBF_CRLB", "Mean_Log10_Cond"]].to_string(index=False))

    print("\n--- Top 3-PLD Configuration (Secondary Study) ---")
    df_3 = df_search[df_search["N_PLD"] == 3].copy()
    print(df_3.head(3)[["Configuration", "Total_PLD_Sum", "Score", "Early_ATT_CRLB", "Mean_ATT_CRLB", "Mean_CBF_CRLB", "Mean_Log10_Cond"]].to_string(index=False))

    # Selected candidate for Phase 4 & 5
    best_4 = df_4.iloc[0]
    print("\n" + "=" * 80)
    print(f"SELECTED OPTIMIZED 4-PLD CONFIGURATION: {best_4['Configuration']}")
    print(f"  Total PLD sum: {best_4['Total_PLD_Sum']:.3f}s (vs Baseline {baseline_sum:.3f}s)")
    print(f"  Early ATT CRLB: {best_4['Early_ATT_CRLB']:.4f}s (vs Baseline {df_base['Early_ATT_CRLB'].values[0]:.4f}s)")
    print(f"  Condition Number: 10^{best_4['Mean_Log10_Cond']:.2f} (vs Baseline 10^{df_base['Mean_Log10_Cond'].values[0]:.2f})")
    print("=" * 80)

if __name__ == "__main__":
    main()

