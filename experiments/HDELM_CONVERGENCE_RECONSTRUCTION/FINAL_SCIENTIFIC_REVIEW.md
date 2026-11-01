# Final Scientific Review: H-DELM Convergence Reconstruction

## Scope and safeguards

This review uses the existing reconstructed datasets, saved-checkpoint
evaluation, controlled-run histories, and reconstruction metadata. No model
was retrained, and no existing model, dataset, checkpoint, or figure was
modified. The only new files are this review and the accompanying
`dataset_distribution_comparison.csv` and `convergence_stability_analysis.csv`.

The finalized `submited h-delm/` package was checked against all 88 file hashes
recorded in `datasets/reconstruction_metadata.json`: all 88 matched, with no
missing or additional files. The reconstructed dataset file hashes also match
the metadata manifest.

## 1. Numerical results cross-check

Values below were checked against the CSV records, not inferred from plots.
Displayed values are rounded; the CSVs retain full precision.

### Frozen original checkpoints

| Target | Training normalized MSE | Validation normalized MSE | Training physical RMSE | Validation physical RMSE | Checkpoint SHA-256 verified |
|---|---:|---:|---:|---:|---|
| CBF | 0.001101341 | 0.001166840 | 3.318646 | 3.415904 | Yes |
| ATT | 0.015736206 | 0.014504782 | 0.313610 | 0.301090 | Yes |

These are from `results/saved_checkpoint_comparable_losses.csv`. The
checkpoint hashes match the packaged copies and the values recorded in the
reconstruction metadata.

### Controlled-run histories

| Target | Epochs | Best epoch | Best training MSE | Best validation MSE | Final-epoch training MSE | Final-epoch validation MSE |
|---|---:|---:|---:|---:|---:|---:|
| CBF | 14 | 4 | 0.001117634 | 0.001185350 | 0.001137357 | 0.001203305 |
| ATT | 19 | 9 | 0.015819083 | 0.014444292 | 0.015821467 | 0.014833216 |

The values, best epochs, and early-stop flags agree with the corresponding
controlled history CSVs and checkpoint metadata. The controlled checkpoint
is the best-validation state, not the last epoch's state.

### Historical-history cross-check

The copied historical CSVs in `HDELM_CONVERGENCE_AUDIT/original_histories/`
contain 37 CBF and 26 ATT epochs. Their validation minima are CBF epoch 27
(0.001173680) and ATT epoch 16 (0.014497164). Their final recorded losses are
CBF train 0.001097439 / validation 0.001174554 and ATT train 0.015776166 /
validation 0.014504866. These agree with the historical values cited in the
reconstruction report to its stated precision. Historical and controlled
losses are not directly interchangeable because their aggregation and
evaluation timing differ.

## 2. Reconstructed cohort distributions

The metadata records 1,000,000 training rows and 100,000 validation rows,
each with four inputs and two physical targets. The training cohort uses
10,000 random parameter-pair draws and the validation cohort uses 1,000.
Because saved labels are float32, the training file contains **9,999 distinct
stored CBF/ATT pairs** after exact-value deduplication (one pair is duplicated
after storage precision is applied); validation contains 1,000. There is no
exact pair overlap between the cohorts. Each parameter pair is represented
at all 100 SNR values.

The target statistics below are computed once per stored parameter pair, not
by treating the repeated SNR rows as independent:

| Target | Cohort | Pairs | Mean | Sample SD | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| CBF (ml/100g/min) | Training | 10,000 draws (9,999 distinct stored pairs) | 49.7122 | 28.8221 | 0.0314 | 99.9977 |
| CBF (ml/100g/min) | Validation | 1,000 | 52.2152 | 28.4119 | 0.0337 | 99.9687 |
| ATT (s) | Training | 10,000 draws (9,999 distinct stored pairs) | 1.7573 | 0.7224 | 0.5000 | 2.9999 |
| ATT (s) | Validation | 1,000 | 1.7647 | 0.7343 | 0.5017 | 2.9951 |

ATT distributions are very similar: the mean difference is 0.00736 s
(validation minus training), Welch p=0.762 and two-sample KS p=0.740. The CBF
validation mean is 2.5031 ml/100g/min higher (Welch p=0.0081; Holm-adjusted
p=0.0162 over the two target mean tests), but the two-sample KS test does not
show a broad distribution difference (p=0.0743). The standardized CBF mean
difference is small (about 0.087 pooled SD).

Both cohorts contain the exact same 100 SNR levels, linearly spaced from 10
to 20, with 10,000 training and 1,000 validation observations at each level.
Thus, validation is not advantaged by a different SNR schedule or more
high-SNR observations.

Raw signal summaries for the four PLDs are:

| PLD (s) | Training mean ± SD | Validation mean ± SD |
|---:|---:|---:|
| 1.525 | 649.26 ± 477.06 | 673.35 ± 476.98 |
| 2.025 | 556.53 ± 343.96 | 584.21 ± 347.11 |
| 2.525 | 432.77 ± 252.56 | 454.47 ± 252.18 |
| 3.025 | 305.57 ± 177.60 | 319.89 ± 175.94 |

For signal comparisons, the inferential unit is the parameter pair: each
pair's signal was averaged across its 100 SNR observations. Validation
pair-mean signals were higher by 24.09, 27.67, 21.70, and 14.31 signal units
at the four PLDs, respectively. Welch mean comparisons with Holm correction
across PLDs are not significant at 1.525 s (p=0.127), and are significant at
2.025 s (p=0.0385), 2.525 s (p=0.0361), and 3.025 s (p=0.0385). However,
pair-level KS comparisons do not support broad distribution shifts after
correction. Row-level signal comparisons are included in the CSV as
descriptive only; their p-values must not be interpreted as if all repeated
SNR rows were independent.

## 3. Why ATT validation MSE is lower

The ATT checkpoint's physical squared error, averaged per parameter pair
over all 100 SNR observations, is 0.098351 s² on training and 0.090655 s² on
validation. This is a numerical validation reduction of 7.83%. That difference
is **not statistically conclusive** at the independent-pair level:

- Welch comparison: p=0.187.
- Mann–Whitney comparison: p=0.094.
- Pair-resampled 95% bootstrap interval for validation-minus-training mean
  squared error: [-0.01867, 0.00439] s², which includes zero.

The validation ATT target distribution is not detectably shifted. Validation
does have a modestly higher mean CBF and higher mean signal at three later
PLDs; these are possible contributors to different estimation difficulty,
but the analysis does not establish that they cause the lower ATT loss.
Accordingly, the evidence supports **a lower observed validation ATT error
in this reconstructed cohort, but not a statistically established easier
validation cohort or a causal explanation for the loss gap**. Equal SNR
coverage and the uncertainty in pair-level error comparisons further argue
against claiming a proven cohort advantage.

## 4. Controlled convergence stability

`convergence_stability_analysis.csv` contains every epoch's training loss,
validation loss, validation-minus-training gap, best-so-far epoch, and
early-stopping flag, plus summary rows.

### CBF

- The minimum is at epoch 4: validation MSE 0.001185350.
- The second-lowest value is epoch 11: 0.001194728, 0.79% above the minimum.
- The final epoch's validation MSE is 0.001203305, 1.51% above the minimum.
- The best-to-final absolute difference is 0.000017955.
- Validation loss exceeds training loss in all 14 epochs; the gap ranges from
  0.00005205 to 0.00012419.

The epoch-4 minimum is a local low amid a visibly fluctuating trajectory;
epoch 3 is materially higher and epoch 5 is already close to the minimum.
This single run does not demonstrate that epoch 4 would remain best under a
different seed or minibatch order.

### ATT

- The minimum is at epoch 9: validation MSE 0.014444292.
- The second-lowest value is epoch 16: 0.014500410, 0.39% above the minimum.
- The final epoch's validation MSE is 0.014833216, 2.69% above the minimum.
- The best-to-final absolute difference is 0.000388923.
- Validation loss is below training loss in all 19 epochs; the gap ranges
  from -0.00159420 to -0.00098825 (mean -0.00122275).

Several ATT epochs near the minimum are close in value, but the final loss
has risen from the best. This is compatible with a relatively flat region
around the selected epoch followed by fluctuation/degradation; it does not
prove seed-level stability. No validation observations were used for
gradient updates.

## Publication Interpretation and Limitations

**A. Historical DNN training curves.** The original curves are the records
associated with the historical fit (CBF best epoch 27; ATT best epoch 16).
They may be described as historical training records only. Training loss was
an equal average of minibatch means while weights changed, whereas validation
loss was measured after epoch updates in evaluation mode; incomplete batches
also received the same weight as full batches. Therefore, the two plotted
curves are not strictly comparable epoch-end estimates. The original arrays
were not persisted, so corrected losses for the historical cohort/model
cannot be reconstructed from this workspace.

**B. Existing saved checkpoint evaluation on reconstructed cohorts.** These
are valid, full-cohort, sample-weighted evaluation results for the frozen
historical best checkpoints on reconstructed data. They are point evaluations,
not per-epoch training curves and not evaluations on the original historical
arrays.

**C. New controlled-run convergence curves.** These use evaluation-mode,
sample-weighted train and validation MSE after every epoch and are
mathematically comparable within the controlled run. They represent a
separate seeded reproducibility experiment on reconstructed cohorts, not the
exact historical optimization trajectory and not the original historical
checkpoint's training curve.

**Publication recommendation:** If the paper needs curves for the actual
historically fitted model, only the historical curves correspond to that
training run, and they require an explicit caveat about their asymmetric
loss calculation. The corrected controlled-run figures can legitimately be
published only as results of the separately labeled controlled
reconstruction. They must not be substituted or captioned as the actual
historical model's training. No fully corrected historical epoch-end curve
can be produced from the persisted artifacts.

## Methods, limitations, and deliverables

The input cohorts were loaded as saved; no generation, fitting, tuning, or
validation-guided model selection was performed in this review. Cohort
comparisons use independent parameter pairs as the inferential unit where
possible. The 100 SNR observations within a pair are repeated measurements,
not 100 independent parameter samples. Mean-comparison p-values are
exploratory, and the reported target and PLD families use Holm adjustments.
The frozen ATT error comparison uses pair-level resampling and is not
evidence that validation is intrinsically easier.

Detailed cohort summaries and comparisons are in
`dataset_distribution_comparison.csv`; epoch-level and summary stability
metrics are in `convergence_stability_analysis.csv`. Existing experiment
files and figures remain unchanged.
