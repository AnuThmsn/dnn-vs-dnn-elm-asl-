# H-DELM convergence reconstruction report

## Scope and safeguards

All new outputs are confined to `experiments/HDELM_CONVERGENCE_RECONSTRUCTION/`.
The original `HDELM_CONVERGENCE_AUDIT/` is preserved, and its original
historical plots were copied to `figures/historical_original_audit/`. The
finalized `submited h-delm/` package, its histories, and existing checkpoints
were not modified. Checkpoint SHA-256 values were checked before and after
evaluation/training and against packaged checkpoint copies. Controlled-run
checkpoints exist only in this reconstruction directory.

## Original audit findings

The initial audit found that historical training loss averages minibatch mean
losses while model weights change, whereas validation loss is measured in
evaluation mode after the epoch updates. Both use the same normalized-MSE
definition, but the history aggregation gives a smaller final batch the same
weight as a full batch. No dropout or batch normalization is present. Best
historical validation epochs were CBF 27 and ATT 16. The original arrays were
not persisted, so the exact historical losses cannot be recomputed on those
arrays.

## Dataset reconstruction

The generator was reconstructed directly from
`experiments/paper_reproduction_snr10_15_20/run_reproduction_pipeline.py`
(`generate_paper_dataset`) and `experiments/physics.py`, without modifying
either source:

- Training: 10,000 unique CBF/ATT pairs × 100 linearly spaced SNRs from 10
  through 20, generated using seed 42 = 1,000,000 rows.
- Validation: 1,000 unique pairs × the same 100 SNR levels, seed 99 =
  100,000 rows.
- Original PLDs, Buxton GKM, dual-channel quadrature Rician noise, and
  physiological parameter bounds were used.
- Input mean and standard deviation were recalculated from reconstructed
  training signals only, then applied to both cohorts.

The data arrays, normalization statistics, metadata, shapes, seeds, and
SHA-256 hashes are in `datasets/`. These are **reconstructed deterministic
cohorts**, not byte-identical historical arrays: the original training and
validation arrays were not saved, and numerical/library/device differences
may also affect generated floating-point values.

## Existing checkpoint evaluation

The original CBF/ATT checkpoints were loaded strictly and evaluated with
`model.eval()` and `torch.no_grad()` on all reconstructed observations.
Normalized MSE was accumulated as total squared error divided by total sample
count; no gradient updates occurred. Target normalization used the same
physical bounds for train and validation.

### CBF

| Evidence source | Train normalized MSE | Validation normalized MSE | Validation below train? |
|---|---:|---:|---|
| Original best checkpoint evaluated consistently on reconstructed cohorts | 0.001101341 | 0.001166840 | No |
| Controlled run at best-validation epoch 4 | 0.001117634 | 0.001185350 | No |
| Controlled run at final epoch 14 | 0.001137357 | 0.001203305 | No |

- Original history best validation epoch: 27; controlled
  best epoch: 4; controlled epochs completed:
  14 (early stop triggered: True).
- Original historical training loss at its last epoch was
  0.001097439; original validation loss was
  0.001174554. Those original training losses were collected
  during updates and are not directly comparable to epoch-end eval losses.
- Controlled validation was below training in 0 of
  14 epochs. Physical-unit RMSE at the best controlled epoch:
  train 3.343103; validation
  3.442891.

### ATT

| Evidence source | Train normalized MSE | Validation normalized MSE | Validation below train? |
|---|---:|---:|---|
| Original best checkpoint evaluated consistently on reconstructed cohorts | 0.015736206 | 0.014504782 | Yes |
| Controlled run at best-validation epoch 9 | 0.015819083 | 0.014444292 | Yes |
| Controlled run at final epoch 19 | 0.015821467 | 0.014833216 | Yes |

- Original history best validation epoch: 16; controlled
  best epoch: 9; controlled epochs completed:
  19 (early stop triggered: True).
- Original historical training loss at its last epoch was
  0.015776166; original validation loss was
  0.014504866. Those original training losses were collected
  during updates and are not directly comparable to epoch-end eval losses.
- Controlled validation was below training in 19 of
  19 epochs. Physical-unit RMSE at the best controlled epoch:
  train 0.314435; validation
  0.300461.


Existing-checkpoint loss values and hashes are in
`results/saved_checkpoint_comparable_losses.csv`. The ATT validation/train
relationship for this reconstructed dataset is therefore the result shown
above; it need not match the historical online-averaged curves. A difference
does not imply that either the historical checkpoint or this evaluation is
incorrect.

## Controlled convergence reconstruction

CBF and ATT were trained separately using the source `DNN_G_Net` architecture:
4 hidden layers × 64 units, ELU activations, sigmoid-bounded target output,
Adam at 1e-3, batch size 512, gradient clipping at 1.0, patience 10, and
normalized MSE. Python, NumPy, CPU/CUDA PyTorch seeds were set to
42 before each target network initialization. Data-generation
seeds remain 42 and 99. Library versions, device, deterministic backend
settings, source hashes, and checkpoint hashes are recorded in
`datasets/reconstruction_metadata.json` and `models/`.

After each epoch's updates, both datasets were evaluated in `model.eval()`
with `torch.no_grad()`. Batch squared-error sums were accumulated and divided
by the exact number of samples, giving sample-weighted MSE. Validation was
never used for gradient updates. Early stopping used the complete validation
MSE; best controlled checkpoints and best epochs are recorded in the
controlled histories and metadata.

The controlled run does not reproduce the original optimizer trajectory:
the historical training arrays were unavailable, and a new seeded
reconstruction run was performed. It is not expected to reproduce the
original best epochs exactly. Besides the non-persisted historical arrays and
the changed loss reporting (epoch-end evaluation instead of online minibatch
loss), the controlled run gives the shuffled DataLoader its own generator
seeded to 42. The original creates the shuffled DataLoader after model
initialization and uses the global PyTorch generator, so minibatch order is
not the same even though the target initialization seed is 42. GPU/library
kernel nondeterminism may also contribute. Architecture and hyperparameters
were not changed; the controlled histories are not asserted to be bitwise
repeatable across hardware/software environments.

## Figures and recommendation

The corrected curves plot controlled-run sample-weighted, eval-mode losses:

- `figures/cbf_corrected_convergence.png` and `.pdf`
- `figures/att_corrected_convergence.png` and `.pdf`
- `figures/combined_corrected_convergence.png` and `.pdf`

Original audit plots remain separately available under
`figures/historical_original_audit/`. The corrected controlled curves are the
appropriate figures to present when describing sample-weighted, end-of-epoch
train and validation losses. Label them as a **controlled reconstruction**,
not the exact historical training trajectory. If the publication specifically
reports the original fitted model, retain its historical curves only with an
explicit caveat about online/unweighted-batch aggregation and missing original
arrays; do not silently substitute the reconstruction for historical
training results.

## Artifacts and reproduction

See `README.md` for the reproduction command. Detailed arrays and hashes are
in `datasets/`; frozen-checkpoint metrics and new per-epoch histories are in
`results/`; controlled best checkpoints are in `models/`.
