# Stage 2: Numerical Scaling and Target Transformations Report

**Experiment:** ASL-MRI DNN-ELM Improvement Campaign — Experiment 3: Reference Values, Numerical Scaling, Calibration, and ELM Correction Under Fixed PLDs  
**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Workspace:** `D:\dnn vs dnn elm`  
**Experiment Directory:** `experiments/reference_calibration_03/`  
**Date:** October 13, 2026  
**Status:** Stage 2 Evaluation Complete  

---

## 1. Executive Summary of Findings

Stage 2 evaluated whether modifying input normalization, latent feature scaling, target standardization, or bounded output transformations could improve CBF and ATT estimation under the fixed four-PLD acquisition protocol ($[1.525, 2.025, 2.525, 3.025]\text{ s}$, $\tau=1.8\text{ s}$). All transformations were fitted strictly on the training partition ($N=100,000$, seed 2026) and evaluated on the held-out validation cohort ($N=100,000$).

### Key Insights:
1. **Target Equivariance Across Affine Scalings:**
   - Standardized target scaling ($(y - \mu)/\sigma$) and Min-Max target scaling ($[-1, 1]$) yielded virtually identical validation metrics (CBF RMSE: **$3.7108\text{ mL/100g/min}$**, $+0.81\%$ gain; ATT RMSE: **$0.3041\text{ s}$**, $+0.06\%$ gain).
   - This mathematically confirms that linear Ridge regression is equivariant under affine target transformations when the L2 regularization parameter $\alpha$ is appropriately scaled.
2. **Failure of Bounded Logit Transformation for ATT:**
   - A smooth logit transformation $z = \operatorname{logit}((y - 0.5)/2.5)$ was tested to enforce strict mathematical bounding within $(0.5, 3.0)\text{ s}$.
   - While it reduced early-ATT RMSE from $0.4589\text{ s}$ to $0.4024\text{ s}$, it caused a **$-3.80\%$ overall regression on ATT** (RMSE degraded from $0.3041\text{ s}$ to **$0.3158\text{ s}$**) and induced a severe negative bias ($-0.0438\text{ s}$). The extreme non-linearity of the sigmoid inverse amplifies prediction errors in the physiological midrange ($1.5 - 2.5\text{ s}$).
3. **Latent Standardization Degrades Representation Quality:**
   - Standardizing penultimate latents reduced the matrix condition number from $16,422$ down to $119$ (CBF) and from $484$ down to $48$ (ATT).
   - However, validation accuracy **degraded**: CBF gain dropped from $+0.81\%$ to $+0.15\%$, and ATT exhibited a $-0.22\%$ regression.
   - **Mechanism:** The DNN encodes feature salience in the relative activation magnitudes across hidden units. Forcing all latents to unit variance artificially amplifies noisy, low-variance latent channels at the expense of primary predictive channels.
4. **Direct Signal Scalings Cannot Match Deep Latents:**
   - Training ELMs directly on 4-PLD signals (bypassing the DNN) achieved CBF RMSE of $3.8891$ and ATT RMSE of $0.3325$ (vs $3.7108$ and $0.3041$ for Feature ELM).
   - Log transformation ($\ln(1 + X)$) slightly improved direct signal ATT RMSE ($0.3205\text{ s}$ vs $0.3325\text{ s}$), but degraded CBF ($3.9320$ vs $3.8891$).

---

## 2. Target Transformations Comparison (Feature ELM, $H=2048$)

| Target | Transformation | Optimal $\alpha$ | Val RMSE | Val MAE | Val Bias | Lin's CCC | $R^2$ | Gain vs DNN % | Early ATT RMSE (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CBF** | **Standardized** ($(y - \mu)/\sigma$) | 10.0 | **3.7108** | 2.4516 | +0.0108 | 0.9916 | 0.9834 | **+0.81%** | — |
| **CBF** | **MinMax $[-1, 1]$** | 10.0 | **3.7108** | 2.4516 | +0.0108 | 0.9916 | 0.9834 | **+0.81%** | — |
| **CBF** | **Raw Physical Units** | 10.0 | 3.7109 | 2.4513 | +0.0108 | 0.9916 | 0.9834 | +0.81% | — |
| **ATT** | **Standardized** ($(y - \mu)/\sigma$) | 1000.0 | **0.3041** | 0.1913 | -0.0028 | 0.9031 | 0.8230 | **+0.06%** | 0.4589 |
| **ATT** | **MinMax $[-1, 1]$** | 1000.0 | **0.3041** | 0.1913 | -0.0028 | 0.9031 | 0.8230 | **+0.06%** | 0.4589 |
| **ATT** | **Raw Physical Units** | 100.0 | 0.3043 | 0.1914 | -0.0030 | 0.9030 | 0.8227 | -0.02% | 0.4585 |
| **ATT** | **Bounded Logit** | 1000.0 | 0.3158 | 0.1971 | -0.0438 | 0.9041 | 0.8090 | **-3.80%** | 0.4024 |

*Baseline A DNN Validation: CBF RMSE = 3.7411 mL/100g/min, ATT RMSE = 0.3042 s.*

---

## 3. Latent Feature Normalization & Conditioning Audit

| Target | Latent Scaling Scheme | Condition Number | Val RMSE | Val MAE | Val Bias | Lin's CCC | Gain vs DNN % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CBF** | **Raw Latents (Baseline)** | 16,422 | **3.7108** | 2.4516 | +0.0108 | 0.9916 | **+0.81%** |
| **CBF** | Standardized Latents ($(F - \mu)/\sigma$) | 119 | 3.7354 | 2.4606 | +0.0141 | 0.9915 | +0.15% |
| **CBF** | Robust Scaled Latents ($(F - \text{med})/\text{IQR}$) | 338 | 3.7305 | 2.4574 | +0.0135 | 0.9916 | +0.28% |
| **ATT** | **Raw Latents (Baseline)** | 484 | **0.3042** | 0.1910 | -0.0027 | 0.9031 | **+0.02%** |
| **ATT** | Standardized Latents ($(F - \mu)/\sigma$) | 48 | 0.3049 | 0.1913 | -0.0026 | 0.9026 | -0.22% |
| **ATT** | Robust Scaled Latents ($(F - \text{med})/\text{IQR}$) | 2,913 | 0.3103 | 0.1958 | -0.0027 | 0.8985 | -1.99% |

---

## 4. Direct Signal-Based Scaling (Without DNN Backbone)

| Signal Scaling Scheme | Input Condition Number | CBF Val RMSE | CBF Val MAE | CBF Val CCC | ATT Val RMSE | ATT Val MAE | ATT Val CCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Z-Score** | 10.57 | 3.8891 | 2.7035 | 0.9908 | 0.3325 | 0.2246 | 0.8811 |
| **Robust Median/IQR** | 10.57 | 3.8891 | 2.7035 | 0.9908 | 0.3325 | 0.2246 | 0.8811 |
| **Log-Scaled ($\ln(1 + X)$)** | 11.33 | 3.9320 | 2.7703 | 0.9906 | 0.3205 | 0.2186 | 0.8907 |
| **Min-Max $[0, 1]$** | 10.57 | 3.8891 | 2.7035 | 0.9908 | 0.3325 | 0.2246 | 0.8811 |

---

## 5. Stage 2 Conclusions & Next Step

1. **Optimal Configuration Frozen:** Standard z-score target standardization and raw penultimate latents represent the mathematically and empirically optimal representation.
2. **Conditioning vs Accuracy Trade-off Disproved:** Lowering condition numbers by artificially normalizing neural latent activations harms downstream regression accuracy.
3. **Stage 2 Completed:** We commit Stage 2 artifacts before proceeding to Stage 3: Physical Reference Value Sensitivity Analysis.
