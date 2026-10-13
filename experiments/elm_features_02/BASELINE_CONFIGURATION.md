# Baseline Configuration & Audit — Experiment 2: Physics-Informed Features & Leakage-Free Residual ELM

**Experiment:** ASL-MRI DNN-ELM Improvement Campaign — Experiment 2  
**Repository:** https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-  
**Experiment Directory:** `experiments/elm_features_02/`  
**Date:** 2026-10-13  
**Status:** Frozen Baseline Established  

---

## 1. Frozen Checkpoints & Provenance

The primary baseline is Baseline A (Experiment G DNN-G), trained on standard 4-PLD synthetic ASL signals. All network weights are strictly frozen and will not be fine-tuned or retrained.

| Component | Path | SHA256 Hash | Input / Output Dims | Architecture |
| :--- | :--- | :--- | :---: | :--- |
| **CBF Network** | `experiments/baseline_A/CBFnet_G_baselineA.pth` | `76154e7d429263a278401c2f038e78bfa05b0090a0d658b2aa1daee95ecda69b` | $4 \to 1$ | 5 hidden layers $\times$ 128 neurons, ELU activations |
| **ATT Network** | `experiments/baseline_A/ATTnet_G_baselineA.pth` | `91688d8beecf5151dfdb4092d16e4d06c910918b0411fc72c7da15f764025016` | $4 \to 1$ | 4 hidden layers $\times$ 64 neurons, ELU activations |
| **Normalization Stats** | `experiments/baseline_A/normalization_stats.npz` | `d2637da7af19559a75862190692017e2a720cfe6b4cebeec2e29b8ceaac8f728` | — | Train mean/std for inputs and targets |
| **Exp 1 Test Metrics** | `experiments/elm_tuning_01/final_test_metrics.csv` | `08bd33d73ca94c56ee9c74ef834fdb0abd98eb7451465bf1722e6d562054e7a6` | — | Locked test evaluation of tuned ELMs |

---

## 2. Feature Extraction Layer & Representation

- **Penultimate Latent Extraction:**
  - For CBF: Activations immediately preceding the linear output projection layer, yielding a **128-dimensional** feature vector $F_{\text{CBF}} \in \mathbb{R}^{128}$.
  - For ATT: Activations immediately preceding the linear output projection layer, yielding a **64-dimensional** feature vector $F_{\text{ATT}} \in \mathbb{R}^{64}$.
- **Mathematical Definition:**
  Given normalized input $X_n \in \mathbb{R}^4$:
  $$h^{(1)} = \text{ELU}(W_1 X_n + b_1)$$
  $$h^{(l)} = \text{ELU}(W_l h^{(l-1)} + b_l), \quad l = 2, \dots, L-1$$
  $$\text{Latent Feature: } F(X_n) = h^{(L-1)}$$
  $$\text{Standardized Output: } \hat{y}_{\text{std}} = W_L F(X_n) + b_L$$

---

## 3. Dataset Splits & Data Invariant Definitions

The evaluation uses identical data splits created during the forensic baseline audit:
1. **Training Partition:** $N = 1,000,000$ synthetic observations (`cache/X_train.npy`, `cache/Y_train.npy`).
2. **Validation Cohort:** $N = 100,000$ held-out synthetic observations (`cache/X_val.npy`, `cache/Y_val.npy`). Used exclusively for feature selection, representation evaluation, and residual model selection.
3. **Locked Test Cohort:** $N = 100,000$ held-out synthetic observations (`cache/X_test.npy`, `cache/Y_test.npy`, `cache/SNR_test.npy`). Quarantined until Stage 5.

### Normalization & Physical Clipping:
- **Input Normalization:**
  $$\mu_X = [656.0311, 560.5862, 435.96695, 308.37015]$$
  $$\sigma_X = [480.29407, 344.71332, 253.01886, 179.04381]$$
- **Target Transformations:**
  - CBF: $\mu_{\text{CBF}} = 50.00099\text{ mL/100g/min}$, $\sigma_{\text{CBF}} = 28.86714\text{ mL/100g/min}$.
    Physical clipping range: $[10.0, 120.0]\text{ mL/100g/min}$.
  - ATT: $\mu_{\text{ATT}} = 1.751554\text{ s}$, $\sigma_{\text{ATT}} = 0.722069\text{ s}$.
    Physical clipping range: $[0.5, 3.0]\text{ s}$.

### Noise & Acquisition Protocol:
- **Acquisition Timing:** Fixed 4-PLD baseline: $[1.525, 2.025, 2.525, 3.025]\text{ s}$, with label duration $\tau = 1.8\text{ s}$.
- **Physiological Constants:** $T_{1t} = 1.2\text{ s}$, $T_{1a} = 1.66\text{ s}$, $\alpha = 0.85$, $\beta = 0.75$, $\lambda = 0.9\text{ mL/g}$.
- **Noise Model:** Dual-channel complex Rician noise:
  $$S_{\text{noisy}} = \sqrt{(S_{\text{clean}} + \epsilon_1)^2 + \epsilon_2^2}, \quad \epsilon_1, \epsilon_2 \sim \mathcal{N}(0, \sigma^2)$$
  $$\sigma = \frac{M_0}{\text{SNR}}$$
  SNR levels sampled uniformly across: $\{5, 8, 10, 12, 15, 20, 30, 50, 100, 200, 500, \infty\}$.

---

## 4. Current Baseline Benchmark Metrics

Below are the audited, reproducible baseline metrics established from saved prediction arrays and Experiment 1:

### A. Validation Set Performance ($N=100,000$):
| Model | CBF Configuration | ATT Configuration | CBF RMSE (mL/100g/min) | CBF MAE | CBF CCC | ATT RMSE (s) | ATT MAE (s) | ATT CCC | Early ATT RMSE (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (DNN-G)** | StandardizedNet (Frozen) | StandardizedNet (Frozen) | 3.741121 | 2.545329 | 0.991481 | 0.304244 | 0.193415 | 0.902785 | 0.465942 |
| **Historical Feature ELM (5 seeds)** | $H=512, \alpha=10^{-2}, \tanh$ | $H=512, \alpha=10^{-2}, \tanh$ | 3.702740 | 2.443725 | 0.991684 | 0.303652 | 0.190683 | 0.903222 | 0.460830 |
| **Exp 1 Tuned ELM (5 seeds)** | $H=2048, \alpha=1.0, \text{elu}$ | $H=2048, \alpha=100.0, \tanh$ | 3.700330 | 2.435969 | 0.991695 | 0.303598 | 0.190602 | 0.903261 | 0.460782 |

### B. Locked Test Set Performance ($N=100,000$):
| Model | CBF Configuration | ATT Configuration | CBF RMSE (mL/100g/min) | CBF MAE | CBF CCC | ATT RMSE (s) | ATT MAE (s) | ATT CCC | Early ATT RMSE (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (DNN-G)** | StandardizedNet (Frozen) | StandardizedNet (Frozen) | 3.726580 | 2.532334 | 0.991566 | 0.305800 | 0.193525 | 0.900921 | 0.466430 |
| **Historical Feature ELM (5 seeds)** | $H=512, \alpha=10^{-2}, \tanh$ | $H=512, \alpha=10^{-2}, \tanh$ | 3.682789 | 2.423241 | 0.991791 | 0.304875 | 0.190390 | 0.901608 | 0.469328 |
| **Exp 1 Tuned ELM (5 seeds)** | $H=2048, \alpha=1.0, \text{elu}$ | $H=2048, \alpha=100.0, \tanh$ | 3.681497 | 2.421046 | 0.991797 | 0.304813 | 0.190454 | 0.901637 | 0.469343 |

---

## 5. Early-ATT Breakdown

In both Baseline A DNN and the tuned Feature ELMs, estimation in the early arrival regime ($[0.5, 1.0)\text{ s}$) suffers from severe degradation:
- Early-ATT RMSE: $\approx 0.466 - 0.469\text{ s}$ (compared to overall RMSE of $\approx 0.305\text{ s}$).
- Early-ATT Mean Bias: $+0.369$ to $+0.372\text{ s}$ (systematic overestimation).
- Lin's CCC in $[0.5, 1.0)\text{ s}$: $\approx 0.005$ (near zero correlation).

This baseline audit is frozen. Experiment 2 will systematically investigate feature representations (Stage 2) and leakage-free residual learning (Stage 3).
