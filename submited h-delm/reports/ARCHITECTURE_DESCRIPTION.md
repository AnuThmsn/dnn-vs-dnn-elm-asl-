# Implemented H-DELM V2 architecture

The evaluated architecture is the repository's residual H-DELM V2
implementation, not a confirmed reproduction of a source paper. No reference
H-DELM paper was available in the workspace.

Each target has a separate, frozen DNN-G model. Its input is four normalized
ASL measurements. The network has four fully connected 64-unit hidden layers
with ELU activations. A sigmoid output is scaled to the physical bounds:
CBF 0–100 ml/100g/min or ATT 0.5–3.0 s.

For each target-specific ELM, the 64-dimensional penultimate DNN activation is
concatenated with 15 signal-derived features (the four measurements; three
neighboring PLD slopes; three neighboring signal ratios; mean, variance,
minimum, maximum; and trapezoidal area). Calibration-set feature mean and
standard deviation are used to standardize the resulting 79 features.

The ELM has 512 fixed random hidden neurons with `tanh` activation. Its
output weights are obtained by ridge regression to predict
`y_true - y_DNN`. In the saved artifacts the ridge parameter is 0.01. The
final estimate is `clip(y_DNN + predicted_residual, physical_bounds)`.

The random hidden projection uses seed 42. The DNN is frozen during ELM
calibration. The ELM is fitted using a separate 100,000-observation
calibration cohort (80,000 fit and 20,000 validation in the configuration).
The saved calibration sweep favored λ=1.0, while the evaluated saved artifacts
use λ=0.01; this inconsistency is retained and documented rather than altered.

