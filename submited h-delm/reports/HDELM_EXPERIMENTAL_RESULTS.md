# H-DELM experimental results

## Architecture and protocol

This package reports the existing H-DELM V2 residual-refinement model against
its frozen DNN-G backbone. Each target has a separate four-hidden-layer,
64-neuron ELU DNN-G network. H-DELM appends 15 signal-derived features to the
DNN penultimate representation, yielding 79 inputs for a 512-unit fixed
random `tanh` ELM. Ridge regression predicts the DNN residual; the correction
is added to the DNN output and clipped to the physiological bounds. The source
paper is unavailable, so paper-level architectural fidelity is not claimed.

The DNN-G pipeline specifies 1,000,000 simulated training observations and
100,000 validation observations generated with the implemented Buxton model,
the four configured PLDs, and dual-channel quadrature Rician noise. Training
data are produced in memory and were not saved with a run digest, so the exact
historical arrays cannot be independently hash-linked to the checkpoint.
H-DELM V2 uses a separate 100,000-observation calibration cohort. The fresh
independent test dataset contains exactly 10,000 observations at each of SNR
10, 15, and 20; both models were run on the same observations. The test seed
is 20261101.

## CBF and ATT results

Values below are measured or calculated from the saved test predictions.
RMSE/MAE use the same units as each parameter; bias is prediction minus truth.

| SNR | Parameter | Model | MSE | RMSE | MAE | Bias | Lin's CCC |
|---:|---|---|---:|---:|---:|---:|---:|
| 10 | CBF | DNN-G | 16.885292 | 4.109172 | 3.223274 | +0.897259 | 0.989487 |
| 10 | CBF | H-DELM V2 | 16.286638 | 4.035671 | 3.160972 | +0.560263 | 0.989882 |
| 10 | ATT | DNN-G | 0.128092 | 0.357899 | 0.232657 | +0.018324 | 0.861893 |
| 10 | ATT | H-DELM V2 | 0.128371 | 0.358289 | 0.233326 | +0.014865 | 0.860951 |
| 15 | CBF | DNN-G | 10.772454 | 3.282142 | 2.462447 | +0.200837 | 0.993552 |
| 15 | CBF | H-DELM V2 | 10.603810 | 3.256349 | 2.437992 | -0.087133 | 0.993668 |
| 15 | ATT | DNN-G | 0.095322 | 0.308743 | 0.202120 | +0.006078 | 0.899215 |
| 15 | ATT | H-DELM V2 | 0.095502 | 0.309035 | 0.202825 | +0.001637 | 0.898684 |
| 20 | CBF | DNN-G | 8.851813 | 2.975200 | 2.166215 | +0.040636 | 0.994690 |
| 20 | CBF | H-DELM V2 | 8.802725 | 2.966939 | 2.165984 | -0.227402 | 0.994731 |
| 20 | ATT | DNN-G | 0.087518 | 0.295834 | 0.192704 | -0.006191 | 0.908470 |
| 20 | ATT | H-DELM V2 | 0.087570 | 0.295922 | 0.192874 | -0.010317 | 0.908099 |

Across all 30,000 observations, CBF RMSE is 3.488532 for DNN-G and 3.449308
for H-DELM V2. ATT RMSE is 0.321938 and 0.322202, respectively. More complete
tables are supplied as CSV under `tables/` and `results/`.

## SNR effect and residual correction

CBF RMSE falls with increasing SNR for both methods. H-DELM V2 has a numerical
CBF RMSE reduction at all three tested SNRs: 1.789% at SNR 10, 0.786% at SNR
15, and 0.278% at SNR 20. ATT RMSE is slightly higher for H-DELM V2 at every
tested SNR (0.109%, 0.094%, and 0.030% relative worsening, respectively).
These are small changes and do not show consistent improvement across both
targets.

The paired residual diagnostic compares the target residual `truth − DNN`
with the ELM-predicted residual, and compares prediction errors before and
after correction. This is descriptive and should not be interpreted as proof
that residual refinement is generally useful. Figures provide the actual
sample-level distributions.

## Statistical analysis

Two-sided paired Wilcoxon signed-rank tests were run on the per-observation
squared errors for each of the six SNR/parameter combinations; p-values were
Holm-adjusted across the six comparisons. CBF improvements are statistically
significant after correction at SNR 10 (adjusted p=3.30e-36) and SNR 15
(adjusted p=5.36e-06), but not at SNR 20 (adjusted p=1.0). ATT comparisons are
not significant after correction. The statistical test evaluates paired
squared-error differences, not a direct test of the RMSE statistic.

Paired percentile bootstrap confidence intervals for RMSE improvement
(DNN-G RMSE minus H-DELM RMSE) are generated from the saved predictions with
5,000 resamples and seed 20261102; see `tables/statistical_comparison.csv`.
Confidence intervals and Wilcoxon tests answer related but distinct
questions.

## Limitations

- The reference H-DELM paper is missing; architecture correspondence cannot
  be asserted.
- The original DNN training arrays were generated in memory and not retained
  with a data hash; the 1M train count is supported by source and history
  artifacts, but not cryptographically linked to checkpoints.
- The DNN-G training uses 10,000 unique physiological parameter pairs repeated
  across 100 SNR levels.
- The calibration sweep's lowest validation RMSE was at λ=1.0, while final
  saved artifacts use λ=0.01.
- Evaluation is on simulated data, not clinical or patient measurements.
- The Bland–Altman analysis describes agreement on simulated samples only.

