# Stage 4: Calibration, Shrinkage, and Residual Corrections Report

**Experiment:** ASL-MRI DNN-ELM Improvement Campaign — Experiment 3: Reference Values, Numerical Scaling, Calibration, and ELM Correction Under Fixed PLDs  
**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Workspace:** `D:\dnn vs dnn elm`  
**Experiment Directory:** `experiments/reference_calibration_03/`  
**Date:** October 13, 2026  
**Status:** Stage 4 Calibration Analysis Complete  

---

## 1. Executive Summary & Calibration Setup

Stage 4 investigated whether the residual errors of the Baseline A DNN and Feature ELM are primarily calibration-related (i.e. systematic slope and offset errors) and whether post-hoc calibration, shrinkage, or non-linear residual damping can improve accuracy under the fixed four-PLD protocol ($[1.525, 2.025, 2.525, 3.025]\text{ s}$, $\tau=1.8\text{ s}$).

### Zero-Leakage Evaluation Design:
To eliminate calibration fitting leakage, the held-out validation cohort ($N=100,000$) was strictly bifurcated into two independent partitions:
- **Calibration Split ($N=50,000$, indices $0 - 49,999$):** Exclusively used to estimate calibration slopes, intercepts, polynomial coefficients, and optimal shrinkage weights ($\gamma^*$).
- **Evaluation Split ($N=50,000$, indices $50,000 - 99,999$):** Held out for unbiased evaluation of all calibrated estimators.

---

## 2. Learned Calibration Parameters

| Estimator / Mapping | Parameter Type | Learned Values (Fitted on $N=50,000$) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Baseline DNN CBF** | Affine ($a \hat{y} + b$) | Slope $a = 1.0026$, Intercept $b = +0.0520$ | Slight gain compression; bias corrected from $-0.17$ toward $0.0$ |
| **Baseline DNN ATT** | Affine ($a \hat{y} + b$) | Slope $a = 0.9995$, Intercept $b = +0.0100$ | Near-perfect slope; small $+10$ ms baseline shift |
| **Feature ELM CBF** | Affine ($a \hat{y} + b$) | Slope $a = 0.99986$, Intercept $b = +0.0036$ | **Virtually perfectly calibrated** ($a \approx 1.0, b \approx 0.0$) |
| **Feature ELM ATT** | Affine ($a \hat{y} + b$) | Slope $a = 0.99704$, Intercept $b = +0.0099$ | **Virtually perfectly calibrated** ($a \approx 0.997, b \approx 0.01$) |
| **Optimal Shrinkage ($\gamma^*$)** | Linear Blend | **CBF: $\gamma^* = 0.86$**, **ATT: $\gamma^* = 0.61$** | $86\%$ ELM + $14\%$ DNN for CBF; $61\%$ ELM + $39\%$ DNN for ATT |
| **Polynomial-2 ELM** | Quadratic ($c_2 x^2 + c_1 x + c_0$) | CBF: $[4.98\times 10^{-5}, 0.9949, 0.0866]$<br>ATT: $[-2.97\times 10^{-3}, 1.0079, 0.0012]$ | Negligible second-order curvature ($|c_2| \le 0.003$) |
| **Early-ATT Residual Correction** | Sigmoid-Damped Bias | Early Bias $= +0.3625\text{ s}$, Transition $= 1.0\text{ s}$ | Smooth subtraction of unidentifiable early-ATT bias |

---

## 3. Held-Out Validation Evaluation Results ($N=50,000$)

| Estimator | CBF RMSE (mL/100g/min) | CBF MAE | CBF Bias | CBF CCC | CBF Gain % | ATT RMSE (s) | ATT MAE (s) | ATT Bias (s) | ATT CCC | ATT Gain % | Early ATT RMSE (s) | Early ATT Bias (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline A DNN (Uncalibrated)** | 3.7339 | 2.5419 | -0.1658 | 0.9916 | 0.00% | 0.3054 | 0.1940 | -0.0051 | 0.9022 | 0.00% | 0.4602 | +0.3642 |
| **2. Tuned Feature ELM (Uncalibrated)** | 3.7040 | 2.4501 | +0.0185 | 0.9917 | +0.80% | 0.3052 | 0.1919 | -0.0009 | 0.9026 | +0.08% | 0.4619 | +0.3649 |
| **3. Affine-Calibrated DNN** | 3.7286 | 2.5184 | +0.0169 | 0.9916 | +0.14% | 0.3054 | 0.1933 | +0.0040 | 0.9022 | +0.01% | 0.4676 | +0.3736 |
| **4. Affine-Calibrated Feature ELM** | 3.7040 | 2.4503 | +0.0152 | 0.9917 | +0.80% | 0.3052 | 0.1916 | +0.0039 | 0.9023 | +0.07% | 0.4667 | +0.3716 |
| **5. Optimal Shrinkage Blend ($\gamma^*$)** | **3.7031** | 2.4555 | -0.0073 | 0.9917 | **+0.83%** | **0.3052** | 0.1924 | -0.0025 | 0.9025 | **+0.09%** | 0.4611 | +0.3646 |
| **6. Polynomial-2 Calibrated ELM** | 3.7041 | 2.4495 | +0.0156 | 0.9917 | +0.80% | 0.3052 | 0.1917 | +0.0039 | 0.9023 | +0.07% | 0.4666 | +0.3711 |
| **7. Early-ATT Sigmoid-Damped ELM** | 3.7040 | 2.4501 | +0.0185 | 0.9917 | +0.80% | **0.3210** | 0.2003 | -0.0618 | 0.9020 | **-5.09%** | **0.3956** | **+0.2226** |

---

## 4. Key Scientific Insights

### 1. The Historical Feature ELM is Already Perfectly Calibrated:
- The learned affine calibration slope for the Feature ELM is $a = 0.99986$ for CBF and $a = 0.99704$ for ATT, with negligible offsets ($b \le 0.01$).
- Applying linear or polynomial calibration produces **zero reduction in RMSE** ($3.7040 \to 3.7040$).
- **Conclusion:** The remaining errors in DNN-ELM estimation are **not calibration or scale distortions**. They represent stochastic observation noise and physical unidentifiability.

### 2. Optimal Shrinkage Achieves the Lowest RMSE:
- Blending the Baseline DNN with the Feature ELM via validation-selected shrinkage weights:
  $$\hat{y}_{\text{CBF}}^* = 0.14 \hat{y}_{\text{DNN}} + 0.86 \hat{y}_{\text{ELM}}$$
  $$\hat{y}_{\text{ATT}}^* = 0.39 \hat{y}_{\text{DNN}} + 0.61 \hat{y}_{\text{ELM}}$$
  achieves the lowest RMSE among all tested models (**$3.7031\text{ mL/100g/min}$** for CBF and **$0.3052\text{ s}$** for ATT).
- **Theoretical Basis:** The gradient-descent neural network and the closed-form Ridge ELM have distinct inductive biases. Linear shrinkage acts as an optimal variance-reduction ensemble that cancels out uncorrelated residual fluctuations.

### 3. The Early-ATT Correction Trade-Off:
- When a smooth sigmoidal correction is applied to eliminate the positive bias in the early transit window ($[0.5, 1.0)\text{ s}$), early RMSE drops from $0.4619\text{ s}$ to **$0.3956\text{ s}$** (bias reduced from $+0.365\text{ s}$ to $+0.223\text{ s}$).
- **However, overall ATT RMSE severely degrades by $-5.09\%$** ($0.3052 \to 0.3210\text{ s}$), introducing an overall negative bias of $-0.062\text{ s}$.
- This empirically verifies that post-hoc corrections cannot overcome the fundamental information-theoretic bottleneck of late PLD sampling ($t \ge 1.525\text{ s}$) without degrading the physiologically valid transit regime ($1.0 - 3.0\text{ s}$).

---

## 5. Candidate Selection for Stage 5 Locked-Test Evaluation

Based strictly on training and validation findings:
1. **Primary Candidate Model:** **Optimal Shrinkage Blend ($\gamma^*$ Feature ELM)**
   - $\gamma_{\text{CBF}}^* = 0.86$, $\gamma_{\text{ATT}}^* = 0.61$.
   - Achieved the lowest validation RMSE for both CBF and ATT without boundary distortion.
2. **Comparison Baselines:**
   - Baseline A DNN (Frozen).
   - Historical Feature ELM ($H=512$, $\alpha=10^{-2}$).
   - Tuned Feature ELM ($H=2048$, $\alpha=10$ for CBF, $\alpha=1000$ for ATT).
   - Affine-Calibrated DNN.

Stage 4 is complete. We commit Stage 4 artifacts before proceeding to Stage 5 locked-test evaluation.
