# Experiment 5 — Stage 2 Fair Final Benchmark Report

**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Working Directory:** `D:\dnn vs dnn elm\experiments\final_results_05`  
**Execution Date:** 2026-10-13  
**Evaluation Cohort:** Locked Test Cohort ($N=100,000$, Fixed 4-PLD pCASL Protocol)  

---

## 1. Executive Summary

Stage 2 reproduces the authoritative benchmark comparison between the frozen **Baseline A DNN-G** and the primary defensible **Feature ELM** candidates (both Single-Seed and 5-Seed Ensemble) on the strictly locked test set ($N=100,000$).

All evaluations strictly follow the established protocol:
- Target physiological ranges: $\text{CBF} \in [0, 100]\text{ mL}/(100\text{g}\cdot\text{min})$, $\text{ATT} \in [0.5, 3.0]\text{ s}$.
- Units: CBF in $\text{mL}/(100\text{g}\cdot\text{min})$; ATT in $\text{seconds}$.
- Improvement sign convention:
  $$\Delta_{\text{RMSE}} = \text{RMSE}_{\text{DNN}} - \text{RMSE}_{\text{Candidate}}$$
  $$\text{Gain}_{\%} = 100 \times \frac{\text{RMSE}_{\text{DNN}} - \text{RMSE}_{\text{Candidate}}}{\text{RMSE}_{\text{DNN}}}$$
  *(Positive indicates candidate improvement; negative indicates candidate degradation).*

---

## 2. Reproduced Benchmark Metrics ($N=100,000$)

| Metric | Baseline A (DNN-G) | Single-Seed Feature ELM (Seed 42) | Ensemble Feature ELM (5 Seeds) |
| :--- | :---: | :---: | :---: |
| **CBF RMSE** ($\text{mL}/(100\text{g}\cdot\text{min})$) | 3.7266 | 3.6876 | **3.6849** |
| **CBF MAE** ($\text{mL}/(100\text{g}\cdot\text{min})$) | 2.5323 | 2.4313 | **2.4296** |
| **CBF Bias** | -0.1731 | -0.0082 | **-0.0080** |
| **CBF CCC** | 0.9916 | 0.9918 | **0.9918** |
| **CBF $R^2$** | 0.9833 | 0.9837 | **0.9837** |
| **CBF $\Delta \text{RMSE}$** | 0.0000 | +0.0390 | **+0.0417** |
| **CBF % Gain** | 0.00% | +1.05% | **+1.12%** |
| **CBF 95% Bootstrap CI** | — | $[+0.0351, +0.0432]$ | $[+0.0378, +0.0457]$ |
| **CBF Paired $t$-test $p$** | — | $5.34 \times 10^{-81}$ | $4.28 \times 10^{-97}$ |
| **ATT RMSE** ($\text{seconds}$) | 0.3058 | 0.3050 | **0.3048** |
| **ATT MAE** ($\text{seconds}$) | 0.1935 | 0.1907 | **0.1905** |
| **ATT Bias** | -0.0066 | -0.0008 | **-0.0008** |
| **ATT CCC** | 0.9009 | 0.9015 | **0.9016** |
| **ATT $R^2$** | 0.8195 | 0.8205 | **0.8207** |
| **ATT $\Delta \text{RMSE}$** | 0.0000 | +0.0008 | **+0.0010** |
| **ATT % Gain** | 0.00% | +0.28% | **+0.31%** |
| **ATT 95% Bootstrap CI** | — | $[+0.00063, +0.00106]$ | $[+0.00075, +0.00115]$ |
| **ATT Paired $t$-test $p$** | — | $3.53 \times 10^{-14}$ | $5.56 \times 10^{-20}$ |
| **Early-ATT RMSE $[0.5, 1.0)\text{s}$** | **0.4664** | 0.4694 | 0.4694 |
| **Early-ATT Bias $[0.5, 1.0)\text{s}$** | **+0.3691** | +0.3717 | +0.3717 |

---

## 3. Stratified Robustness and Defensibility Analysis

### 3.1 Did the Candidate Improve CBF?
**Yes, modestly.**  
The Single-Seed Feature ELM reduced CBF RMSE from $3.7266$ to $3.6876\text{ mL}/(100\text{g}\cdot\text{min})$ ($+1.05\%$ gain, $\Delta = 0.0390$), and the 5-seed ensemble further reduced it to $3.6849$ ($+1.12\%$ gain, $\Delta = 0.0417$). The paired 95% bootstrap confidence interval strictly excludes zero ($[+0.0378, +0.0457]$). CBF bias was substantially reduced from $-0.1731$ to $-0.0080$.

### 3.2 Did the Candidate Improve ATT?
**Negligibly.**  
The Single-Seed Feature ELM reduced ATT RMSE from $0.3058$ to $0.3050\text{ s}$ ($+0.28\%$ gain, $\Delta = 0.0008\text{ s}$), and the 5-seed ensemble achieved $0.3048\text{ s}$ ($+0.31\%$ gain, $\Delta = 0.0010\text{ s}$). While statistically significant due to $N=100,000$ ($p = 5.56 \times 10^{-20}$), an absolute error reduction of $1\text{ millisecond}$ ($0.001\text{ s}$) has zero practical significance in clinical MRI.

### 3.3 Did the Candidate Improve Both Parameters Simultaneously?
**Yes.**  
Both CBF and ATT show positive percentage improvements simultaneously without trade-offs. However, the magnitude for ATT is an order of magnitude smaller than CBF.

### 3.4 Robustness Across Acquisition SNR Levels
Both models track each other closely across all 12 SNR levels:
- **Low SNR ($5.0\text{ dB}$):**
  - CBF RMSE: DNN = $9.0747$, ELM = $9.0560$ ($+0.21\%$ gain)
  - ATT RMSE: DNN = $0.5050$, ELM = $0.5042$ ($+0.16\%$ gain)
- **High SNR ($50.0\text{ dB}$):**
  - CBF RMSE: DNN = $0.6970$, ELM = $0.6409$ ($+8.05\%$ gain)
  - ATT RMSE: DNN = $0.1017$, ELM = $0.1009$ ($+0.79\%$ gain)
- *Conclusion:* The Feature ELM provides modest benefits at high SNR where signal fidelity allows precise latent hyperplane fitting, but does not provide special robustness under severe noise.

### 3.5 Early-ATT Estimation ($[0.5, 1.0)\text{ s}$)
**Degraded.**  
In the early arrival window ($\text{ATT} < 1.0\text{ s}$), the Feature ELM worsens estimation accuracy:
- Baseline DNN RMSE: **0.4664 s**, Bias: **+0.3691 s**
- Feature ELM RMSE: **0.4694 s**, Bias: **+0.3717 s**
- *Physical Cause:* Because the first PLD is $1.525\text{ s}$, any bolus arriving before $1.0\text{ s}$ has completely cleared its inflow phase before the first measurement. The downstream regularized readout shrinks uncertain estimates toward the prior distribution mean ($1.75\text{ s}$), slightly increasing positive bias and error for short arrival times.
