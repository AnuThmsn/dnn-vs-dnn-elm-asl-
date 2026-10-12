# Experiment 1: Leakage-Free ELM Optimization — Protocol & Baseline Specification

**Experiment Directory:** `experiments/elm_tuning_01/`  
**Date:** October 13, 2026  
**Primary Baseline:** Baseline A (Experiment G DNN-G) and Ensemble Feature ELM (5 seeds)  
**Acquisition:** Fixed Reference 4-PLD Schedule $[1.525, 2.025, 2.525, 3.025]	ext{ s}$, $\tau = 1.8	ext{ s}$  

---

## 1. Objective and Hypothesis
The objective of Experiment 1 is to evaluate whether systematic hyperparameter optimization of the Extreme Learning Machine (ELM) readout head can improve parameter estimation accuracy over the existing hand-selected configuration ($H=512, \alpha=0.01, \text{activation}=\text{tanh}$) on frozen DNN feature representations.

**Non-negotiable Scientific Controls:**
1. **Zero Architecture Drift:** The underlying DNN feature extractors (`CBFnet_G_baselineA.pth` and `ATTnet_G_baselineA.pth`) remain strictly frozen.
2. **Fixed Acquisition Protocol:** PLDs are held constant at $[1.525, 2.025, 2.525, 3.025]	ext{ s}$.
3. **Zero Test Set Leakage:** Hyperparameters ($H, \alpha, \text{activation}$) are tuned and selected strictly on the validation set ($N=100,000$). The locked test set ($N=100,000$) is evaluated exactly once at the conclusion.
4. **Decoupled Optimization:** CBF and ATT hyperparameters are tuned independently.

---

## 2. Frozen Feature Extractor and Normalization Specifications
- **CBF Backbone:** `experiments/baseline_A/CBFnet_G_baselineA.pth`  
  - Architecture: StandardizedNet (4 inputs $\to$ 5 hidden layers $\times$ 128 neurons $\to$ ELU).
  - Penultimate Latent Dimension: $d = 128$.
  - SHA-256: `76154e7d429263a278401c2f038e78bfa05b0090a0d658b2aa1daee95ecda69b`
- **ATT Backbone:** `experiments/baseline_A/ATTnet_G_baselineA.pth`  
  - Architecture: StandardizedNet (4 inputs $\to$ 4 hidden layers $\times$ 64 neurons $\to$ ELU).
  - Penultimate Latent Dimension: $d = 64$.
  - SHA-256: `91688d8beecf5151dfdb4092d16e4d06c910918b0411fc72c7da15f764025016`
- **Target Normalization:**
  - CBF Mean / Std: $50.0010 / 28.8671$
  - ATT Mean / Std: $1.7516 / 0.7221$

---

## 3. Verified Baseline Performance (Reproduced from Saved Test Predictions, $N=100,000$)

| Model | CBF RMSE | CBF MAE | CBF CCC | ATT RMSE (s) | ATT MAE (s) | ATT CCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline A (DNN-G)** | 3.726580 | 2.532334 | 0.991566 | 0.305800 | 0.193525 | 0.900921 |
| **Ensemble Feature ELM (5 seeds)** | 3.682789 | 2.423241 | 0.991791 | 0.304875 | 0.190390 | 0.901608 |

---

## 4. Hyperparameter Search Space (Validation Set Tuning)
- **Hidden Neuron Counts ($H$):** $128, 256, 512, 1024, 2048$
- **Activation Functions:** $\text{tanh}$, $\text{elu}$, $\text{relu}$
- **Ridge Regularization ($lpha$):** Logarithmic grid $[10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0]$
- **Multi-Seed Stability:** 10 random seeds $[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]$ for candidate finalists.
- **Selection Rule:** Primary objective is minimum Validation RMSE, with MAE and CCC monitored.
