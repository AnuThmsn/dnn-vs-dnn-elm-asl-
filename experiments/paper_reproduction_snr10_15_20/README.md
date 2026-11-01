# Independent Reproduction & Controlled Evaluation: DNN-G vs. Proposed Residual H-DELM

**Paper Title:** *A Residual Hybrid DNN–ELM Model for CBF and ATT Estimation in Multi-Delay ASL MRI*  
**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Workspace:** `D:\dnn vs dnn elm`  
**Experiment Directory:** `experiments/paper_reproduction_snr10_15_20/`  
**Date:** November 1, 2026  
**Auditor:** Antigravity Autonomous Research Assistant  

---

## 1. Executive Summary

This study independently reproduces and rigorously evaluates the two primary architectures from the research paper:
1. **DNN-G Baseline:** 4 hidden layers $\times$ 64 neurons for both CBF and ATT with bounded sigmoid output transformations.
2. **Proposed Residual H-DELM:** Frozen 4x64 DNN backbones + 64-dimensional penultimate features + 15 signal-derived physics features (79 total dimensions) feeding a 512-neuron Extreme Learning Machine (ELM) that analytically learns the residual errors of the DNN.

Testing was conducted on a **completely independent simulated cohort of 30,000 observations** (10,000 unique CBF–ATT pairs each at SNR 10, 15, and 20, seed 2026).

---

## 2. Forensic Audit Findings & Code Provenance

A comprehensive repository audit ([`REPRODUCTION_AUDIT.md`](file:///D:/dnn%20vs%20dnn%20elm/experiments/paper_reproduction_snr10_15_20/REPRODUCTION_AUDIT.md)) was conducted before execution:

- **Original Files Reused:**
  - `experiments/physics.py`: Contains the verified Buxton General Kinetic Model (`asl_signal`, `compute_physical_signals`) and dual-channel quadrature Rician noise model (`add_rician_noise`, reference calibration $\text{SIG\_REF\_SC} \approx 334.2039$).
- **Reconstructed Code:**
  - `models/dnn_g.py`: Implements the paper's exact 4x64 backbone for both CBF and ATT with bounded sigmoid output heads ($y_{\min} + (y_{\max} - y_{\min}) \cdot \sigma(z)$) and He/Kaiming normal initialization.
  - `models/residual_hdelm.py`: Implements the exact 15 signal-derived physics features (4 raw PLDs, 3 slopes, 3 ratios, mean, variance, min, max, trapezoidal AUC), standardizes the 79-dimensional concatenation on training data, and solves the analytical ridge regression on target residuals ($r = y_{\text{true}} - \hat{y}_{\text{DNN}}$).
  - `simulation/generate_independent_test.py`: Generates 30,000 independent test samples (10k at SNR 10, 10k at SNR 15, 10k at SNR 20) with unique ground truth parameters sampled from $\mathcal{U}[0, 100]$ (CBF) and $\mathcal{U}[0.5, 3.0]$ (ATT) and seed 2026.
  - `evaluation/calculate_metrics.py`: Computes population-consistent MSE, RMSE, MAE, Lin's Concordance Correlation Coefficient (CCC), and percentage gains.
  - `evaluation/statistical_analysis.py`: Performs 1,000 paired bootstrap resamples for 95% confidence intervals and paired Wilcoxon signed-rank tests.
  - `generate_figures.py`: Generates all 9 publication-quality figures at 300 DPI PNG and vector PDF format.
- **Pretrained Checkpoint Status & Retraining Necessity:**
  - The repository's baseline checkpoint `experiments/baseline_A/CBFnet_G_baselineA.pth` has 5 layers $\times$ 128 neurons (272 kB), which contradicts the paper's 4x64 specification.
  - Therefore, faithful paper reproduction strictly required retraining the paper-matched 4x64 CBF and ATT architectures using the paper's training dataset ($1{,}000{,}000$ samples, seed 42) and validation dataset ($100{,}000$ samples, seed 99).

---

## 3. Simulation & Acquisition Configuration

- **Post-Labeling Delays (PLDs):** $[1.525, 2.025, 2.525, 3.025]$ seconds
- **Labeling Duration ($\tau$):** $1.8$ seconds
- **Physical Constants:** $T_{1,\text{tissue}} = 1.2$ s, $T_{1,\text{blood}} = 1.66$ s, $\alpha = 0.85$, $\beta = 0.75$, $\lambda = 0.9$ mL/g, Scale = $100{,}000.0$
- **Reference SNR Point:** $\text{CBF}=50.0$, $\text{ATT}=1.6$, $\text{PLD}=2.0$, $\text{LD}=1.8 \implies \text{SIG\_REF\_SC} \approx 334.2039$
- **Rician Noise Mechanism:** Two independent complex Gaussian noise channels ($e_1, e_2, e_3, e_4 \sim \mathcal{N}(0, \sigma^2)$ where $\sigma = \text{SIG\_REF\_SC} / \text{SNR}$) yielding $X = M_c + M_l$.

---

## 4. Model Architectures & Hyperparameters

### A. DNN-G Baseline
- **Input Dimension:** 4 (standardized PLD measurements)
- **Architecture:** Linear(4, 64) $\to$ ELU $\to$ Linear(64, 64) $\to$ ELU $\to$ Linear(64, 64) $\to$ ELU $\to$ Linear(64, 64) $\to$ ELU $\to$ Linear(64, 1)
- **Output Head:**
  - $\text{CBF} = 0.0 + (100.0 - 0.0) \times \sigma(z)$
  - $\text{ATT} = 0.5 + (3.0 - 0.5) \times \sigma(z)$
- **Loss:** Normalized MSE: $\frac{1}{B} \sum (\sigma(z) - y_{\text{norm}})^2$
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, batch size 512, max epochs 200, patience 10, grad clip 1.0)
- **Training Outcomes:**
  - CBFnet: Converged at Epoch 37 (Best Val Loss: $0.001174$, Best Val $\text{RMSE} = 3.4259\text{ mL/100g/min}$).
  - ATTnet: Converged at Epoch 26 (Best Val Loss: $0.014497$, Best Val $\text{RMSE} = 0.3010\text{ s}$).

### B. Proposed Residual H-DELM
- **Feature Extraction:**
  - 64 penultimate-layer activations from frozen DNN-G
  - 15 signal-derived physics features: 4 raw signals, 3 slopes, 3 ratios, mean, variance, min, max, trapezoidal AUC
  - Combined feature dimension: $64 + 15 = 79$
- **Standardization:** Zero-mean, unit-variance fitted strictly on training data
- **ELM Hidden Layer:** $H = 512$ random neurons, activation $\tanh$, weights $W, b \sim \mathcal{U}(-1/\sqrt{79}, 1/\sqrt{79})$
- **Target:** Residual error $r = y_{\text{true}} - \hat{y}_{\text{DNN}}$
- **Ridge Regression:** $\beta = (H^T H + \lambda I)^{-1} H^T r$ with $\lambda = 0.01$ in float64 precision
- **Final Prediction:** $\hat{y}_{\text{HDELM}} = \text{clip}(\hat{y}_{\text{DNN}} + \hat{r}, [y_{\min}, y_{\max}])$

---

## 5. Comprehensive Results Table (30,000 Independent Test Observations)

| SNR | Model | Parameter | MSE | RMSE | MAE | Bias | Lin's CCC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | DNN-G | CBF | 17.1556 | 4.1419 | 3.2444 | +0.8502 | 0.9894 |
| **10** | **H-DELM** | **CBF** | **16.5131** | **4.0636** | **3.1710** | **+0.5258** | **0.9898** |
| **10** | **DNN-G** | **ATT** | **0.1252** | **0.3538** | 0.2311 | +0.0266 | **0.8655** |
| **10** | H-DELM | ATT | 0.1254 | 0.3542 | **0.2310** | **+0.0237** | 0.8648 |
| **15** | DNN-G | CBF | 10.2887 | 3.2076 | 2.4249 | +0.2712 | 0.9939 |
| **15** | **H-DELM** | **CBF** | **10.1264** | **3.1822** | **2.3956** | **-0.0197** | **0.9940** |
| **15** | DNN-G | ATT | 0.0937 | 0.3061 | 0.2026 | **-0.0034** | 0.8986 |
| **15** | **H-DELM** | **ATT** | **0.0933** | **0.3054** | **0.2020** | -0.0058 | **0.8989** |
| **20** | DNN-G | CBF | 8.7286 | 2.9544 | 2.1501 | **+0.0347** | 0.9947 |
| **20** | **H-DELM** | **CBF** | **8.6970** | **2.9491** | **2.1473** | -0.2464 | **0.9948** |
| **20** | DNN-G | ATT | 0.0867 | 0.2944 | 0.1929 | **-0.0083** | 0.9090 |
| **20** | **H-DELM** | **ATT** | **0.0864** | **0.2940** | **0.1918** | -0.0105 | **0.9091** |

---

## 6. Percentage Improvement & Paired Bootstrap Significance

Improvement is defined as $\frac{\text{Metric}_{\text{DNN}} - \text{Metric}_{\text{HDELM}}}{\text{Metric}_{\text{DNN}}} \times 100$ (positive indicates H-DELM superiority). Paired bootstrap testing uses 1,000 resamples (seed 42).

| SNR | Parameter | RMSE Improvement (%) | $\Delta_{\text{RMSE}}$ 95% CI | Bootstrap $p$-value | MAE Improvement (%) | $\Delta_{\text{MAE}}$ 95% CI | Wilcoxon $p$-value |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | **CBF** | **+1.89%** | $[+0.0689, +0.0884]$ | **$p < 0.001$** | **+2.26%** | $[+0.0646, +0.0822]$ | $p < 10^{-60}$ |
| **10** | **ATT** | **-0.12%** | $[-0.0010, +0.0002]$ | $p = 0.196$ (NS) | **+0.05%** | $[-0.0003, +0.0006]$ | $p = 0.994$ (NS) |
| **15** | **CBF** | **+0.79%** | $[+0.0153, +0.0346]$ | **$p < 0.001$** | **+1.21%** | $[+0.0215, +0.0375]$ | $p < 10^{-11}$ |
| **15** | **ATT** | **+0.24%** | $[+0.0001, +0.0014]$ | **$p = 0.028$** | **+0.28%** | $[+0.0002, +0.0009]$ | $p = 0.00014$ |
| **20** | **CBF** | **+0.18%** | $[-0.0041, +0.0150]$ | $p = 0.266$ (NS) | **+0.13%** | $[-0.0045, +0.0105]$ | $p = 0.787$ (NS) |
| **20** | **ATT** | **+0.14%** | $[-0.0002, +0.0011]$ | $p = 0.192$ (NS) | **+0.57%** | $[+0.0007, +0.0014]$ | $p < 10^{-17}$ |

*NS = Not Statistically Significant.*

---

## 7. Key Scientific Findings & Discussion

1. **CBF Estimation Gains at Moderate-to-Low SNR:**
   - At $\text{SNR} = 10$, Residual H-DELM provides its largest performance gain: **$+1.89\%$ RMSE improvement** ($4.1419 \to 4.0636\text{ mL/100g/min}$, $p < 0.001$) and **$+2.26\%$ MAE improvement** ($3.2444 \to 3.1710\text{ mL/100g/min}$, $p < 10^{-60}$).
   - At $\text{SNR} = 15$, H-DELM maintains a statistically significant gain: **$+0.79\%$ RMSE improvement** ($p < 0.001$).
   - At $\text{SNR} = 20$, the benefit diminishes to $+0.18\%$ RMSE and is statistically indistinguishable from the baseline ($p = 0.266$), as the DNN-G backbone already approaches the Rician noise error floor.
2. **ATT Estimation Robustness:**
   - At $\text{SNR} = 10$, H-DELM exhibits a negligible numerical difference ($-0.12\%$ RMSE, $p = 0.196$), showing that noise variance at low SNR limits additional transit-time decoding from post-labeling slopes.
   - At $\text{SNR} = 15$, H-DELM achieves a statistically significant improvement (**$+0.24\%$ RMSE**, $95\%\ \text{CI}\ [+0.0001, +0.0014]$, $p = 0.028$).
   - At $\text{SNR} = 20$, MAE improves significantly by **$+0.57\%$** ($p < 10^{-17}$).
3. **Research Integrity & Negative Results:**
   - These findings confirm that while H-DELM provides modest, statistically significant advantages for CBF at low/moderate SNRs, it does not achieve dramatic double-digit improvements on ill-posed 4-PLD pCASL signals. At low SNR, transit-time estimation remains fundamentally constrained by the sampling delays ($1.525 - 3.025$ s) and Rician measurement noise.

---

## 8. Artifact and Publication Figures Index

All generated artifacts are preserved locally in `experiments/paper_reproduction_snr10_15_20/`:
- **Results Tables:**
  - `results/results_snr_10.csv`
  - `results/results_snr_15.csv`
  - `results/results_snr_20.csv`
  - `results/complete_metrics_comparison.csv`
  - `results/percentage_improvement_summary.csv`
  - `results/statistical_bootstrap_comparison.csv`
- **Saved Models & Checkpoints:**
  - `models/cbf_dnn_g_best.pth`
  - `models/att_dnn_g_best.pth`
  - `results/input_normalization_stats.npz`
- **Raw Predictions (30,000 observations):**
  - `results/predictions/independent_test_30k.npz`
  - `results/predictions/reproduction_predictions_30k.npz`
- **Publication Figures (PNG 300 DPI & Vector PDF):**
  - `figures/01_cbf_rmse_vs_snr.png` (.pdf)
  - `figures/02_att_rmse_vs_snr.png` (.pdf)
  - `figures/03_cbf_mae_vs_snr.png` (.pdf)
  - `figures/04_att_mae_vs_snr.png` (.pdf)
  - `figures/05_cbf_ccc_vs_snr.png` (.pdf)
  - `figures/06_att_ccc_vs_snr.png` (.pdf)
  - `figures/07_cbf_prediction_scatter.png` (.pdf)
  - `figures/08_att_prediction_scatter.png` (.pdf)
  - `figures/09_residual_error_distributions.png` (.pdf)
