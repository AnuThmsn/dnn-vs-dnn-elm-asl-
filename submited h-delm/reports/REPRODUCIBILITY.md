# Reproducibility

## Packaged inputs

- `results/predictions.npz`: test signals, ground truth, SNR, DNN-G estimates,
  H-DELM V2 estimates, ELM residual outputs, PLDs, and test seed.
- `models/model_metadata/*_dnn_g_training_history.csv`: actual epoch-by-epoch
  training and validation losses.
- `models/dnn_checkpoint/`: frozen CBF and ATT checkpoints.
- `models/hdelm_parameters/`: ELM random projections, ridge output weights,
  feature scaling, and artifact metadata.
- `datasets/residual_calibration_100k.npz`: V2 residual calibration cohort.
- `datasets/dataset_metadata.json` and `dataset_statistics.csv`: experiment
  generation and distribution summary. The full in-memory DNN training cohort
  was not persisted.
- `scripts/source_*.py`: snapshots of the simulator and model definitions
  required by the evaluation script.

## Regeneration

From the project root (Python 3.10; NumPy, PyTorch, pandas, SciPy, Matplotlib):

```powershell
python "submited h-delm\scripts\reproduce_results.py"
```

The command regenerates the same test cohort using seed 20261101, evaluates
the packaged frozen checkpoints/artifacts, regenerates CSV tables, computes
paired RMSE-bootstrap intervals with 5,000 resamples using seed 20261102, and
writes figures as 600-DPI PNG plus vector PDF and SVG. It uses no test-set
fitting or tuning. Existing source experiment folders are not required by the
script; all model and simulator source used at runtime is copied into
`scripts/source_*.py`.

Individual commands are `evaluate_models.py`, `generate_tables.py`, and
`generate_figures.py` from that folder. Metric definitions (MSE, RMSE, MAE,
prediction-minus-truth bias, Lin's CCC) and the paired bootstrap definition
are implemented in `scripts/publication_utils.py`.

The package records the generating test seed and software versions in its
results audit. Re-running with different library or hardware versions may
cause small floating-point differences.

