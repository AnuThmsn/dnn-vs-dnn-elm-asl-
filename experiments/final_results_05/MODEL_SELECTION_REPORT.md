# Experiment 5 — Stage 3 Model Selection & Controlled Improvement Report

**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Working Directory:** `D:\dnn vs dnn elm\experiments\final_results_05`  
**Execution Date:** 2026-10-13  
**Status:** Completed Controlled Final Improvement Attempt  

---

## 1. Executive Summary & Hypotheses Evaluated

In Stage 3, two low-complexity, mathematically justified improvement attempts were evaluated under strictly leakage-free conditions (fitted on $N_{\text{train}} = 1,000,000$, tuned on $N_{\text{val}} = 100,000$, evaluated on $N_{\text{test}} = 100,000$):

1. **Candidate 1: Optimal Variance-Shrinkage Convex Ensemble Blend**
   - *Hypothesis:* The DNN exhibits slightly better early-ATT stability ($0.4572\text{ s}$ vs $0.4608\text{ s}$ on validation), whereas the Feature ELM achieves lower overall MSE. A convex combination:
     $$\hat{y}_{\text{blend}} = (1 - w) \cdot \hat{y}_{\text{DNN}} + w \cdot \hat{y}_{\text{ELM}}$$
     with $w \in [0, 1]$ tuned on validation data could potentially blend the benefits of both.
   - *Validation Tuning:* Optimal validation weights were found at $w = 0.99$ for CBF and $w = 0.92$ for ATT.
   - *Validation Outcome:* Yielded identical CBF RMSE ($3.7027$) and an imperceptible $0.0001\text{ s}$ ATT RMSE gain ($0.3036$ vs $0.3037$).
   - *Locked Test Set:* Test CBF RMSE was $3.6849$ (identical to Historical ELM $3.6849$); Test ATT RMSE was $0.3049$ (vs $0.3048$ for Historical ELM). Early-ATT RMSE improved by only $0.0003\text{ s}$ ($0.4691$ vs $0.4694$).
   
2. **Candidate 2: Deterministic Latent Linear Ridge Readout**
   - *Hypothesis:* The penultimate representation $h_{\text{penult}} \in \mathbb{R}^{d}$ is already linearly organized by the base DNN. Bypassing the random ELM projection layer and directly solving $L_2$-regularized linear ridge regression:
     $$\beta^* = (F^T F + \alpha I)^{-1} F^T Y$$
     would eliminate random seed variance while preserving the linear subspace advantage.
   - *Validation Tuning:* Optimal $\alpha = 10.0$ for CBF, $\alpha = 100.0$ for ATT.
   - *Validation Outcome:* Achieved CBF RMSE $3.7087$ (+0.87% gain over DNN) and ATT RMSE $0.3040$ (+0.09% gain over DNN).
   - *Locked Test Set:* Test CBF RMSE was $3.6926$ (+0.91% gain) and Test ATT RMSE was $0.3054$ (+0.14% gain).

---

## 2. Quantitative Comparison Table

| Candidate Name | Validation CBF RMSE | Validation CBF Gain | Validation ATT RMSE | Validation ATT Gain | Locked Test CBF RMSE | Locked Test CBF Gain | Locked Test ATT RMSE | Locked Test ATT Gain | Locked Test Early ATT RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (DNN-G)** | 3.7411 | 0.00% | 0.3042 | 0.00% | 3.7266 | 0.00% | 0.3058 | 0.00% | **0.4664** |
| **Historical Feature ELM (5 seeds)** | **3.7027** | **+1.03%** | 0.3037 | +0.19% | **3.6849** | **+1.12%** | **0.3048** | **+0.31%** | 0.4694 |
| **Candidate 1: Convex Ensemble Blend** | **3.7027** | **+1.03%** | **0.3036** | **+0.20%** | **3.6849** | **+1.12%** | 0.3049 | +0.30% | 0.4691 |
| **Candidate 2: Latent Linear Ridge** | 3.7087 | +0.87% | 0.3040 | +0.09% | 3.6926 | +0.91% | 0.3054 | +0.14% | 0.4705 |

---

## 3. Decision and Model Selection

In strict accordance with the experiment specification:
> *"If neither candidate improves the validation results convincingly, stop experimentation and select the strongest existing model. A scientifically valid negative result is preferable to a misleading improvement."*

**Conclusion:**
Neither Candidate 1 nor Candidate 2 produced a meaningful improvement over the **Historical 5-Seed Ensemble Feature ELM**:
- Candidate 1 merely recreated the Feature ELM predictions ($99\%$ and $92\%$ ELM weights).
- Candidate 2 confirmed that direct linear readout captures $80\text{--}90\%$ of the ELM gain, but does not surpass the random feature mapping.

Therefore, no further architectural modifications are warranted. We officially freeze the **Ensemble Feature ELM (5 seeds)** as the definitive, strongest validated DNN+ELM candidate for Stage 4 and Stage 5 presentation package.
