# H-DELM experimental submission package

This folder packages verified results for the latest available H-DELM V2
independent test, compares it with the same frozen DNN-G backbone, and
provides scripts to reproduce tables and figures.

## Main results

- Fresh independent test: 30,000 simulated observations, 10,000 each at SNR
  10, 15, and 20; test seed 20261101.
- DNN-G and H-DELM V2 predictions use the exact same observations.
- CBF RMSE was lower for H-DELM V2 at all three SNRs; paired tests with
  Holm correction support significant improvement at SNR 10 and 15, not 20.
- ATT RMSE was slightly higher for H-DELM V2 at all SNRs; differences were
  not statistically significant after correction.
- The H-DELM reference paper is missing from the workspace. Paper-level
  architecture correspondence is not verified.
- The DNN training cohort is specified as 1M and passed to the DNN training
  loops, but the historical in-memory arrays were not persisted with hashes.
- The final saved H-DELM artifacts use ridge λ=0.01 although the calibration
  sweep favored λ=1.0.

## Contents

- `figures/png`, `figures/pdf`, `figures/svg`: 11 scientific figures in each
  requested format.
- `tables`: architecture, experiment configuration, CBF, ATT, overall, and
  statistical CSV tables.
- `results`: final metrics, comparison, predictions, audit, statistical
  tests, and execution log. Historical V1 and seed-2026 V2 outputs are kept
  under separate `v1_reference/` and `v2_reference/` subfolders.
- `models`: frozen DNN checkpoints, H-DELM ELM artifacts, normalization,
  histories, and V2 configuration.
- `datasets`: V2 calibration data, histories, metadata, and distribution
  summaries. The full historical DNN training arrays were not persisted.
- `scripts`: model/simulator snapshots and scripts to evaluate and regenerate
  tables and figures.
- `reports`: results, architecture, captions, reproducibility, and
  limitations, including V1/V2 provenance.

Run the complete package regeneration from the repository root:

```powershell
python "submited h-delm\scripts\reproduce_results.py"
```
