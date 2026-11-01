# Objective 3 — Final independent SNR test

**Status: VERIFIED WITH QUALIFICATIONS.**

## Execution

A new cohort was generated for this run with seed **20261101**, separate from
the documented DNN training seed 42, validation seed 99, ELM calibration seed
31415, and prior test seed 2026. It contains exactly 10,000 observations at
each of SNR 10, 15, and 20 (30,000 total), with input shape `(30000, 4)` and
target shape `(30000, 2)`. The simulator is the repository's `compute_physical_signals`
and dual-channel `add_rician_noise` from `experiments/physics.py`.

The test-label audit found zero exact CBF/ATT pair intersections with
regenerated training and validation parameter pairs or stored calibration
labels. This is a deterministic overlap check, not a cryptographic proof that
no historical copy of a sample was ever used. No test observations were used
for feature scaling, calibration, tuning, or model selection in this run.

Inference used the V1 `cbf_dnn_g_best.pth` and `att_dnn_g_best.pth` checkpoints,
the training-only input normalization statistics, and the persisted V2
H-DELM artifacts. Both models ran on the identical generated observations.
The H-DELM artifact contains the ELM random `W`, `b`, ridge readout `beta`, and
the 79-feature scaler. SHA-256 hashes and audit details are in
`FINAL_TEST_DATA_AUDIT.json`.

The evaluated ELM artifacts store `ridge_lambda=0.01`. The saved calibration
validation sweep reached its lowest RMSE at `1.0` for both targets, but the
existing V2 driver fits its final artifacts with a hard-coded `0.01`. The
artifacts were evaluated exactly as saved; this is a qualification, not a
test-set adjustment.

The first execution reached inference and metrics but stopped during plot
creation because the overall row was included in a three-SNR chart. The
comparison plot was corrected to exclude the overall row and the complete run
was repeated successfully. No model or hyperparameter was changed.

## RMSE comparison

Improvement is `(DNN-G RMSE − H-DELM RMSE) / DNN-G RMSE × 100`; positive values
favor H-DELM.

| SNR | Parameter | DNN-G RMSE | H-DELM RMSE | Relative RMSE improvement (%) |
|---:|---|---:|---:|---:|
| 10 | CBF | 4.109172 | 4.035671 | +1.788705 |
| 10 | ATT | 0.357899 | 0.358289 | -0.108901 |
| 15 | CBF | 3.282142 | 3.256349 | +0.785847 |
| 15 | ATT | 0.308743 | 0.309035 | -0.094310 |
| 20 | CBF | 2.975200 | 2.966939 | +0.277662 |
| 20 | ATT | 0.295834 | 0.295922 | -0.029726 |
| All (30,000) | CBF | 3.488532 | 3.449308 | +1.124366 |
| All (30,000) | ATT | 0.321938 | 0.322202 | -0.082148 |

## Full metrics

Bias is prediction minus truth. CCC is Lin's concordance correlation
coefficient. The CSV has full precision.

| SNR | Parameter | Model | MSE | RMSE | MAE | Bias | Lin's CCC |
|---:|---|---|---:|---:|---:|---:|---:|
| 10 | CBF | DNN-G | 16.885292 | 4.109172 | 3.223274 | +0.897259 | 0.989487 |
| 10 | CBF | H-DELM | 16.286638 | 4.035671 | 3.160972 | +0.560263 | 0.989882 |
| 10 | ATT | DNN-G | 0.128092 | 0.357899 | 0.232657 | +0.018324 | 0.861893 |
| 10 | ATT | H-DELM | 0.128371 | 0.358289 | 0.233326 | +0.014865 | 0.860951 |
| 15 | CBF | DNN-G | 10.772454 | 3.282142 | 2.462447 | +0.200837 | 0.993552 |
| 15 | CBF | H-DELM | 10.603810 | 3.256349 | 2.437992 | -0.087133 | 0.993668 |
| 15 | ATT | DNN-G | 0.095322 | 0.308743 | 0.202120 | +0.006078 | 0.899215 |
| 15 | ATT | H-DELM | 0.095502 | 0.309035 | 0.202825 | +0.001637 | 0.898684 |
| 20 | CBF | DNN-G | 8.851813 | 2.975200 | 2.166215 | +0.040636 | 0.994690 |
| 20 | CBF | H-DELM | 8.802725 | 2.966939 | 2.165984 | -0.227402 | 0.994731 |
| 20 | ATT | DNN-G | 0.087518 | 0.295834 | 0.192704 | -0.006191 | 0.908470 |
| 20 | ATT | H-DELM | 0.087570 | 0.295922 | 0.192874 | -0.010317 | 0.908099 |
| All | CBF | DNN-G | 12.169853 | 3.488532 | 2.617312 | +0.379577 | 0.992615 |
| All | CBF | H-DELM | 11.897725 | 3.449308 | 2.588316 | +0.081909 | 0.992797 |
| All | ATT | DNN-G | 0.103644 | 0.321938 | 0.209160 | +0.006070 | 0.890138 |
| All | ATT | H-DELM | 0.103814 | 0.322202 | 0.209675 | +0.002062 | 0.889538 |

## Paired statistical tests

Two-sided paired Wilcoxon signed-rank tests were run on per-observation
squared errors for each of the six SNR/parameter comparisons. Holm correction
was applied across all six tests at family-wise alpha 0.05.

| SNR | Parameter | Raw p-value | Holm-adjusted p-value | Significant after correction? |
|---:|---|---:|---:|---|
| 10 | CBF | 5.4982e-37 | 3.2989e-36 | Yes |
| 10 | ATT | 0.626595 | 1.000000 | No |
| 15 | CBF | 1.0720e-06 | 5.3598e-06 | Yes |
| 15 | ATT | 0.308927 | 1.000000 | No |
| 20 | CBF | 0.890523 | 1.000000 | No |
| 20 | ATT | 0.633134 | 1.000000 | No |

The RMSE reduction for CBF is statistically significant at SNR 10 and 15,
but not at SNR 20. Although CBF RMSE is numerically lower at SNR 20, it is not
supported as a significant improvement by this paired test. ATT RMSE is
slightly higher (worse) with H-DELM at all three SNRs; none of those differences
is statistically significant after correction.

## Output files

- `FINAL_METRICS.csv` — MSE, RMSE, MAE, bias, and CCC per SNR/model/parameter,
  including the overall 30K set.
- `FINAL_PER_SNR_COMPARISON.csv` — required RMSE comparison and percentage
  improvements, including overall.
- `FINAL_PREDICTIONS.npz` — exact test signals, labels, SNR, DNN-G and H-DELM
  predictions, residual corrections, seed, and PLDs.
- `FINAL_TEST_DATA_AUDIT.json` — generation, counts, overlap checks, runtime,
  and SHA-256 artifact provenance.
- `PAIRED_STATISTICAL_TESTS.csv` — six paired tests and Holm-adjusted results.
- `PER_SNR_RMSE_COMPARISON.png`, `CBF_PREDICTION_VS_GROUND_TRUTH.png`,
  `ATT_PREDICTION_VS_GROUND_TRUTH.png`, `CBF_RESIDUAL_DISTRIBUTIONS.png`,
  and `ATT_RESIDUAL_DISTRIBUTIONS.png`.
- `EXECUTION_LOG.txt` and `run_final_verification.py`.
