# V1, V2, and final-test provenance

The primary result in `reports/HDELM_EXPERIMENTAL_RESULTS.md` is the fresh
independent test generated with seed **20261101**, evaluated using the frozen
DNN-G checkpoints and saved H-DELM V2 artifacts. DNN-G and H-DELM predictions
in `results/predictions.npz` use exactly the same 30,000 observations.

The package also preserves historical results for context, but they are
separated from the primary analysis:

- `results/v1_reference/` contains the older V1 residual H-DELM outputs and
  their prior 30K independent test (seed 2026).
- `results/v2_reference/` contains the previous V2 three-model comparison and
  V2 predictions on that historical seed-2026 cohort.
- `models/hdelm_parameters/` contains the persisted, calibrated V2 ELM
  artifacts used for the fresh primary result. V1's fitted ELM weights were
  not persisted as a standalone artifact.

The seed-2026 reference results are not merged into the fresh seed-20261101
metrics or figures. V1 fits residuals using DNN training observations; V2
uses an independent calibration cohort. The primary metrics and figures use
V2 only.

