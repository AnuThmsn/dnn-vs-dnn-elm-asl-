# Hybrid Architecture Research

This directory contains a new, isolated research run. The finalized
`submited h-delm/` submission package and all V1/V2 experiment artifacts are
read-only inputs and were not modified.

## Frozen inputs and protocol

- DNN-G checkpoints, normalization statistics, H-DELM V2 artifacts, and the
  100,000-observation calibration cohort are loaded from `submited h-delm/`.
- The existing 30,000-observation independent test archive is used unchanged
  as a fixed benchmark. It was already used in the earlier DNN-G/H-DELM report,
  so it is independent of the calibration cohort but is **not a previously
  unseen test for this new research phase**. No new model settings are chosen
  from its results.
- New ELM ablations and the residual MLP are fitted from a deterministic
  80,000/20,000 split of the calibration cohort. Feature scalers are fitted
  only on the calibration training split. The MLP uses the calibration
  validation split for early stopping.
- The attached `results/BEST_RESULTS_PRESENTATION/tables/snr_stratified_metrics.csv`
  is a different historical evaluation (different SNR grid and per-SNR counts)
  and is not pooled with this fixed 30K benchmark.

## Completed scope

`scripts/run_core_comparisons.py` reproduces the frozen DNN-G and saved H-DELM
V2 baselines, runs representation/physics feature ablations and residual
versus direct-output ELM comparisons, selects residual correction strengths
using calibration validation only, and trains a frozen-backbone residual MLP.
The combined residual ELM and residual MLP were each run with three seeds.
Generated models, logs, metrics, and plots are kept under this directory.

Remaining research includes broader multi-seed validation, kernel and
multi-head models, additional MLP configurations, extensive feature-group
ablations, and reduced-PLD retraining. The completed three-seed comparison is
limited to the combined residual ELM and the 128→64 residual MLP; it does not
establish generalization across DNN training seeds. The other phases are not
claimed complete. See `reports/RESEARCH_FINDINGS.md` for actual results and
remaining work.
