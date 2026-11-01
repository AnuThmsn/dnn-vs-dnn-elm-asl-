# H-DELM convergence reconstruction

This is an isolated reconstruction of the DNN-G training and validation
convergence experiment. It does not modify the finalized `submited h-delm/`
package, the original training histories, the initial
`HDELM_CONVERGENCE_AUDIT/`, or any existing checkpoint.

## Reproduce

From the repository root, run:

```powershell
python "experiments\HDELM_CONVERGENCE_RECONSTRUCTION\run_reconstruction.py"
```

The script reconstructs train/validation cohorts with the original simulator
and data seeds, writes their hashes and train-only input normalization,
evaluates the saved checkpoints without gradients, then trains separate CBF
and ATT DNNs with the audited architecture and optimizer settings. At every
epoch end, it evaluates all train and validation observations in evaluation
mode and computes sample-weighted normalized MSE. Best controlled checkpoints
are written only under this experiment's `models/`.

The original cohorts were not persisted, so reconstructed arrays cannot be
claimed byte-identical to the historical training arrays. The controlled
training histories are a reproducibility run, not a reconstruction of the
exact original optimization trajectory.

## Output layout

- `datasets/`: reconstructed compressed cohorts, normalization, and metadata.
- `models/`: new controlled-run checkpoints and metadata only.
- `results/`: existing-checkpoint comparable losses and controlled histories.
- `figures/`: corrected controlled-run plots plus copied historical plots.
- `CONVERGENCE_RECONSTRUCTION_REPORT.md`: comparison and limitations.
