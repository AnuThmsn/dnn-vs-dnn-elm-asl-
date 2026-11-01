# Paper Reproduction Experiment (Version 2)
## A Residual Hybrid DNN–ELM Model for CBF and ATT Estimation in Multi-Delay ASL MRI

---

## 1. Overview and Scientific Motivation

This directory (`experiments/paper_reproduction_snr10_15_20_v2/`) contains the scientifically corrected reproduction and evaluation of the **DNN-G baseline** and the proposed **Residual Hybrid DNN–ELM (H-DELM)** model for arterial spin labeling (ASL) cerebral blood flow (CBF) and arterial transit time (ATT) quantification.

### Version Differences:
- **V1 (`experiments/paper_reproduction_snr10_15_20/`):** Baseline reproduction where the residual ELM was fitted using the 1,000,000 observations from the DNN training cohort.
- **V2 (`experiments/paper_reproduction_snr10_15_20_v2/`):** Corrected reproduction adhering to the paper's theoretical framework, where the residual ELM is calibrated using a completely independent, held-out cohort of **100,000 simulated observations** (`seed 31415`). Both V1 and V2 models are evaluated against the DNN-G baseline on the **exact same 30,000 independent test observations** (`seed 2026`).

---

## 2. Directory Layout

```
experiments/paper_reproduction_snr10_15_20_v2/
├── config/
│   └── experiment_config_v2.json           # Master JSON configuration for V2
├── models/
│   ├── residual_hdelm_v2.py                # H-DELM V2 model with save/load functionality
│   ├── cbf_hdelm_v2_artifact.npz           # Serialized CBF ELM weights, projections & bounds
│   └── att_hdelm_v2_artifact.npz           # Serialized ATT ELM weights, projections & bounds
├── simulation/
│   └── generate_calibration_dataset.py     # Independent 100k calibration generator (seed 31415)
├── evaluation/
│   ├── calculate_metrics_v2.py             # MSE, RMSE, MAE, Bias, Lin's CCC, and Gain
│   └── statistical_testing_v2.py           # Paired permutation, bootstrap & Holm-Bonferroni
├── results/
│   ├── data/
│   │   └── residual_calibration_100k.npz   # 100k independent calibration signals and targets
│   ├── predictions/
│   │   └── reproduction_predictions_30k_v2.npz # 3-way test predictions (DNN, V1, V2)
│   └── tables/
│       ├── complete_3way_metrics_comparison.csv
│       ├── percentage_improvement_summary_v2.csv
│       ├── statistical_hypothesis_testing_v2.csv
│       ├── lambda_sensitivity_cbf.csv
│       └── lambda_sensitivity_att.csv
├── figures/                                # Publication figures (PNG 300 DPI + Vector PDF)
│   ├── 01_dnn_residual_distribution_train_vs_cal.[png|pdf]
│   ├── 02_hdelm_correction_distribution_cal_vs_test.[png|pdf]
│   ├── 03_residual_error_before_after_correction.[png|pdf]
│   ├── 04_cbf_prediction_scatter_3way.[png|pdf]
│   ├── 05_att_prediction_scatter_3way.[png|pdf]
│   ├── 06_error_vs_ground_truth_parameter.[png|pdf]
│   ├── 07_error_vs_snr.[png|pdf]
│   ├── 08_lambda_sensitivity_curve.[png|pdf]
│   └── 09_v1_vs_v2_percentage_gain_comparison.[png|pdf]
├── logs/
│   └── execution_v2.log                    # Complete execution log
├── standalone_inference.py                 # Self-contained zero-retraining inference engine
├── generate_diagnostics_figures_v2.py      # Diagnostic and publication visualization script
├── run_reproduction_pipeline_v2.py         # End-to-end automated reproduction pipeline
├── REPRODUCTION_AUDIT.md                   # Formal audit report for V2
└── README.md                               # This file
```

---

## 3. How to Run Standalone Inference

The standalone inference engine allows loading the frozen backbones and ELM artifacts to perform inference on any 4-PLD signal array without retraining:

```python
from standalone_inference import StandaloneASLEstimator
import numpy as np

# Initialize estimator (loads DNN checkpoints and ELM artifacts automatically)
estimator = StandaloneASLEstimator()

# Raw multi-PLD signals (N, 4) at [1.525, 2.025, 2.525, 3.025] s
X_raw = np.array([
    [750.0, 620.0, 480.0, 320.0],
    [410.0, 360.0, 270.0, 190.0]
], dtype=np.float32)

# Predict CBF and ATT
predictions = estimator.predict(X_raw)

print("Estimated CBF (DNN-G) :", predictions["cbf_dnn"])
print("Estimated CBF (H-DELM):", predictions["cbf_hdelm"])
print("Estimated ATT (DNN-G) :", predictions["att_dnn"])
print("Estimated ATT (H-DELM):", predictions["att_hdelm"])
```

