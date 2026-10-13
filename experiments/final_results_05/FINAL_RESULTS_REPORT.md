# Experiment 5 — Stage 4 Final Model Selection Report

**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Working Directory:** `D:\dnn vs dnn elm\experiments\final_results_05`  
**Execution Date:** 2026-10-13  
**Status:** Complete & Authoritative  

---

## 1. Executive Summary & Selection Philosophy

In Stage 4, the definitive model selections for the project's final scientific presentation were established. Selection was conducted strictly on the independent validation split ($N_{\text{val}} = 100,000$), balancing estimation accuracy, computational runtime, memory overhead, reproducibility, and physical validity. The locked test cohort ($N_{\text{test}} = 100,000$) was evaluated only after model freezing.

### Core Selections:
1. **Primary Baseline Model:** Frozen **Baseline A (DNN-G)** (`CBFnet_G_baselineA.pth`, `ATTnet_G_baselineA.pth`).
2. **Primary Defensible DNN+ELM Model:** **5-Seed Ensemble Feature ELM** ($H=512$, $\alpha=10^{-2}$, $\tanh$, penultimate features Rep A).
3. **Reference Benchmarks Included for Context:**
   - *Single-Seed Feature ELM (Seed 42):* Validates single-model performance without ensembling.
   - *High-Capacity Tuned Feature ELM ($H=2048$, 5 seeds):* Represents the upper-bound capacity benchmark from Experiment 1.
   - *Deterministic Latent Linear Ridge Readout:* Demonstrates the performance of direct linear regularized mapping without random feature expansion.

---

## 2. Authoritative Final Results Table

Evaluated on the locked test cohort ($N=100,000$):

| Model Category | Model Name | Architecture | CBF RMSE | CBF MAE | CBF % Gain | CBF Paired 95% CI | ATT RMSE | ATT MAE | ATT % Gain | ATT Paired 95% CI | Early ATT RMSE $[0.5, 1.0)\text{s}$ | Parameter-Specific Limitations |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Primary Frozen Baseline** | **Baseline A (DNN-G)** | StandardizedNet (5-layer CBF, 4-layer ATT) | 3.7266 | 2.5323 | 0.00% | $[+0.0000, +0.0000]$ | 0.3058 | 0.1935 | 0.00% | $[+0.00000, +0.00000]$ | **0.4664** | Negative CBF bias ($-0.173\text{ mL}/(100\text{g}\cdot\text{min})$); slightly higher baseline variance. |
| **Primary Defensible DNN+ELM** | **Ensemble Feature ELM (5 seeds)** | Penultimate Features + 5-Seed ELM ($H=512$, $\alpha=10^{-2}$) | **3.6849** | **2.4296** | **+1.12%** | $[+0.0378, +0.0457]$ | **0.3048** | **0.1905** | **+0.31%** | $[+0.00075, +0.00115]$ | 0.4694 | Early-ATT error slightly degraded ($+0.003\text{ s}$); ATT gain is clinically imperceptible ($1\text{ ms}$). |
| **Reference Single-Seed ELM** | Single-Seed Feature ELM (Seed 42) | Penultimate Features + Single ELM ($H=512$, $\alpha=10^{-2}$) | 3.6876 | 2.4313 | +1.05% | $[+0.0351, +0.0432]$ | 0.3050 | 0.1907 | +0.28% | $[+0.00063, +0.00106]$ | 0.4694 | Single random weight seed variance ($+0.0027\text{ mL}/(100\text{g}\cdot\text{min})$ vs ensemble). |
| **Reference High-Capacity ELM** | Tuned Feature ELM ($H=2048$, 5 seeds) | Penultimate Features + Tuned ELM ($H=2048$, $\alpha=1.0/100$) | 3.6815 | 2.4210 | +1.21% | $[+0.0412, +0.0491]$ | 0.3048 | 0.1905 | +0.32% | $[+0.00078, +0.00118]$ | 0.4693 | 4x larger matrix dimension and memory for marginal $0.003\text{ mL}/(100\text{g}\cdot\text{min})$ gain. |
| **Reference Linear Readout** | Deterministic Latent Linear Ridge | Penultimate Features + Direct Ridge Regression | 3.6926 | 2.4458 | +0.91% | $[+0.0306, +0.0376]$ | 0.3054 | 0.1913 | +0.14% | $[+0.00033, +0.00059]$ | 0.4705 | Captures $80\%$ of ELM gain without random units, but slightly higher RMSE than ELM. |

*Note: Percentage gains are defined as $100 \times (\text{RMSE}_{\text{DNN}} - \text{RMSE}_{\text{Candidate}}) / \text{RMSE}_{\text{DNN}}$. Positive values indicate candidate improvement.*

---

## 3. Honest Scientific Interpretation

### 3.1 What the Feature ELM Does Improve
1. **CBF Estimation:** The Feature ELM consistently reduces CBF RMSE by $\approx 1.12\%$ ($0.0417\text{ mL}/(100\text{g}\cdot\text{min})$) and reduces MAE from $2.5323$ to $2.4296$. More importantly, it almost completely resolves the base DNN's negative systematic bias (from $-0.1731$ to $-0.0080$).
2. **Computational Speed:** Fitting the regularized ELM readout takes only $12\text{--}18\text{ seconds}$ on $1,000,000$ samples, avoiding hours of iterative gradient descent retraining.

### 3.2 What the Feature ELM Does Not Improve
1. **ATT Estimation:** The overall ATT improvement is $+0.31\%$, corresponding to an absolute difference of $0.0010\text{ seconds}$ ($1\text{ millisecond}$). This difference is clinically meaningless in MRI.
2. **Early Bolus Arrival ($[0.5, 1.0)\text{ s}$):** Early-ATT estimation error actually worsens slightly from $0.4664\text{ s}$ to $0.4694\text{ s}$. The fixed 4-PLD acquisition ($[1.525, 2.025, 2.525, 3.025]\text{ s}$) does not record early transit signals; downstream regularized readouts shrink uncertain predictions toward the prior mean, which slightly increases error for short arrival times.
