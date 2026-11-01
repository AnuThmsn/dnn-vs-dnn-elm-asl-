# Objective 2 — DNN training-data verification

**Status: VERIFIED WITH QUALIFICATIONS.**

## Result

The V1 H-DELM DNN-G training pipeline specifies, generates, and passes exactly
**1,000,000 simulated observations** to each DNN's training loop. This was
independently regenerated from the checked-in generator for a non-destructive
statistics audit. The saved training histories and DNN-G checkpoints are
consistent with completed training from that pipeline.

Qualification: V1 constructs the 1M cohort in memory and does not save that
exact cohort, a digest of it, or a run manifest embedded in the checkpoints.
Thus, the code and matching output artifacts support actual 1M training, but
there is no immutable dataset-to-checkpoint hash proving which exact arrays
produced the checkpoint. The separate `experiments/cache/X_train.npy` is also
1M observations, but it belongs to `dataset.py` / `train_baseline_A.py`; it is
not the V1 4×64 DNN-G pipeline's persisted training set and must not be used as
evidence for that checkpoint.

## Dataset generation and audit

The actual H-DELM DNN-G pipeline calls
`generate_paper_dataset(n_unique=10000, n_snr_levels=100, seed=42)`, yielding
`10,000 × 100 = 1,000,000` rows. Validation is generated separately with
`1,000 × 100 = 100,000` rows and seed 99 (`run_reproduction_pipeline.py`,
`generate_paper_dataset`, lines 40–62; training/validation calls lines 84–85).
The same 10,000 CBF/ATT parameter pairs are repeated at each of 100 SNR values;
there are one million noisy signal observations, but only 10,000 unique
physiological parameter pairs in the training cohort.

| Audit item | Observed/reproduced evidence |
|---|---|
| Training `X` / `Y` dimensions | `(1,000,000, 4)` signals and `(1,000,000, 2)` CBF/ATT labels. |
| Validation dimensions | `(100,000, 4)` signals and `(100,000, 2)` labels. Generated with a separate seed and passed to validation. |
| CBF training distribution | min `0.031442`, max `99.997704`, mean `49.712151`, SD `28.820650` ml/100g/min. |
| ATT training distribution | min `0.500033`, max `2.999868`, mean `1.757338`, SD `0.722321` seconds. |
| Training SNR distribution | 100 linearly spaced levels from 10 to 20 inclusive; exactly 10,000 rows per level. No SNR outside that range in this pipeline. |
| Simulated signals | Regenerated signals were finite; signal min `2.910026`, max `2056.354248`, mean `486.031097`, SD `356.453064` (scaled model units). |
| Dataset generation method | CBF and ATT sampled from configured bounds; signals generated with the implemented Buxton GKM and four PLDs; dual-channel quadrature Rician magnitude noise added. |
| Split protocol | Independent seeded generation: train seed 42; validation seed 99. Input mean/std fitted on train only and then applied to validation. |

The simulator constants and ranges are in `experiments/physics.py` (PLDs
line 10; CBF/ATT bounds lines 20–23); the GKM signal equation is
`asl_signal` lines 35–54 and scaling is in `compute_physical_signals` lines
57–73. The dual-channel Rician construction is `add_rician_noise` lines
81–109. This verifies consistency with the repository's implemented Buxton
model and intended noise method; the absent paper prevents a paper-level
confirmation of the intended protocol.

## Training execution, losses, and checkpoints

`run_reproduction_pipeline.py` generates the two cohorts, normalizes inputs
from training statistics only (lines 89–103), and supplies all rows to
`train_dnn_g` for each output (CBF lines 114–123; ATT lines 128–137).
`train_dnn_g` constructs its data loaders from the supplied arrays, shuffles
training rows, keeps validation separate, and saves each best state dictionary
to the named checkpoint. Epoch histories are written to CSV by the same
pipeline.

| DNN | Epochs completed | Final epoch train loss | Final epoch validation loss | Best validation loss (epoch) | Checkpoint (filesystem modified time) |
|---|---:|---:|---:|---:|---|
| CBF | 37 | 0.0010974386 | 0.0011745536 | 0.0011736799 (27) | `experiments/paper_reproduction_snr10_15_20/models/cbf_dnn_g_best.pth` — 2026-11-01 02:19:33 |
| ATT | 26 | 0.0157761656 | 0.0145048658 | 0.0144971643 (16) | `experiments/paper_reproduction_snr10_15_20/models/att_dnn_g_best.pth` — 2026-11-01 02:29:53 |

Corresponding saved histories are
`experiments/paper_reproduction_snr10_15_20/results/cbf_dnn_g_training_history.csv`
and `.../att_dnn_g_training_history.csv`. Each history and its checkpoint have
the same recorded filesystem modification time. The paper-reproduction V2
execution log records loading these V1 checkpoints before running the separate
ELM calibration. Checkpoint hashes and calibration artifact hashes for the
final inference are in `FINAL_TEST_DATA_AUDIT.json`.

The V2 calibration sweep's saved validation tables select `lambda=1.0` for
both CBF and ATT, but `run_reproduction_pipeline_v2.py` instantiates the
final readouts with `ridge_lambda=1e-2`. The evaluated model therefore uses
the persisted `0.01` parameter, not the sweep minimum. This does not change
the DNN's 1M training evidence, but it qualifies the advertised calibration
hyperparameter-selection procedure.

The model's target normalization is min-max scaling to `[0,1]`, and the reported
losses are normalized MSE, not physical-unit MSE (`models/dnn_g.py`,
`train_dnn_g`, lines 94–118). Final validation losses are the values at the
last completed epochs; the selected checkpoint may be from an earlier best
validation epoch as recorded above.

## Provenance limitations

The checked-in `experiments/cache/X_train.npy`, `Y_train.npy`, and
`snr_train.npy` each have 1M entries and are paired with 100K validation
arrays. Their `dataset.py` generator samples SNR from a 12-category mixture
and its separate `train_baseline_A.py` model uses CBF 5×128 / ATT 4×64 and
z-score targets. That is a different experiment from the 4×64 DNN-G
checkpoints used by H-DELM. The V1 H-DELM pipeline instead generates its
10–20 SNR continuous grid in memory and records training histories, not the
full input arrays.

Accordingly: **the 1M simulated-observation count and its use by the
training-loop source are verified, with qualified historical provenance**.
The exact historical in-memory arrays cannot be hash-compared to the checkpoint
because neither the arrays nor their run digest were persisted.
