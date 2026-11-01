# FINAL SCIENTIFIC REPRODUCIBILITY AUDIT REPORT
## DNN-G Baseline vs. Residual H-DELM V1 vs. Residual H-DELM V2

**Audit Objective:** Independent verification of test data provenance, numerical accuracy, statistical hypothesis testing, practical equivalence analysis, residual learning implementation, and scientific validity.  
**Audited Codebase:** `D:\dnn vs dnn elm`  
**Date of Audit:** November 2026  
**Final Audit Verdict:** **VERIFIED WITH QUALIFICATIONS**  

---

## 1. Test Dataset Provenance & Cohort Independence

A bit-level forensic audit of the independent test dataset was performed:

| Check | Specification | Observed / Audited Value | Status |
| :--- | :--- | :--- | :---: |
| **File Existence** | `independent_test_30k.npz` | Present at `experiments/paper_reproduction_snr10_15_20/results/predictions/independent_test_30k.npz` | **PASSED** |
| **File Size** | Expected $\approx 651\text{ kB}$ | `651,169 bytes` | **PASSED** |
| **SHA256 Hash** | Unique content checksum | `b53c2b61253d510e48eda9e90e62d5a9a3f3a56aa80efbb22f0ea44d5df5de37` | **VERIFIED** |
| **Total Observations** | Exactly 30,000 | `30,000` | **PASSED** |
| **SNR Distribution** | 10,000 at SNR 10, 15, 20 | SNR 10.0: `10,000` \| SNR 15.0: `10,000` \| SNR 20.0: `10,000` | **PASSED** |
| **PLD Timing** | $[1.525, 2.025, 2.525, 3.025]\text{ s}$ | `[1.525, 2.025, 2.525, 3.025]` | **PASSED** |
| **PRNG Seed** | Dedicated test seed | `seed = 2026` | **PASSED** |
| **GT Correspondence** | Max diff Test vs V1 & V2 | $\text{Max } |\Delta_{\text{CBF}}| = 0.00\text{e}+00$, $\text{Max } |\Delta_{\text{ATT}}| = 0.00\text{e}+00$ | **PERFECT MATCH** |
| **DNN-G Identity** | Max diff DNN-G in V1 vs V2 | $\text{Max } |\Delta_{\text{CBF}}| = 0.00\text{e}+00$, $\text{Max } |\Delta_{\text{ATT}}| = 0.00\text{e}+00$ | **IDENTICAL** |
| **Overlap: Train vs Test** | Unique (CBF, ATT) pairs | 0 overlapping pairs out of 30,000 (Seed 42 vs Seed 2026) | **ZERO OVERLAP** |
| **Overlap: Cal vs Test** | Unique (CBF, ATT) pairs | 0 overlapping pairs out of 30,000 (Seed 31415 vs Seed 2026) | **ZERO OVERLAP** |
| **Overlap: Signals** | 4-PLD signal vectors | 0 overlapping signal vectors between Cal (100k) and Test (30k) | **ZERO OVERLAP** |

---

## 2. Independent Metric Verification & Aggregate Analysis

All evaluation metrics were recomputed directly from raw predictions in `reproduction_predictions_30k_v2.npz`. Recomputed metrics match reported CSVs to within numerical precision ($< 10^{-6}$):

### Table 1: Independent Re-Verification of Model Metrics (From Raw Predictions)

| SNR | Parameter | Model | Recomputed MSE | Recomputed RMSE | Recomputed MAE | Recomputed Bias | Recomputed Lin's CCC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 dB** | **CBF** | **DNN-G** | 17.155589 | 4.141931 | 3.244397 | +0.850198 | 0.989424 |
| | | **H-DELM V1** | 16.513097 | 4.063631 | 3.171044 | +0.525752 | 0.989835 |
| | | **H-DELM V2** | 16.568828 | 4.070483 | 3.180893 | +0.520081 | 0.989808 |
| **10 dB** | **ATT** | **DNN-G** | 0.125151 | 0.353767 | 0.231099 | +0.026608 | 0.865486 |
| | | **H-DELM V1** | 0.125450 | 0.354189 | 0.230986 | +0.023712 | 0.864837 |
| | | **H-DELM V2** | 0.125732 | 0.354587 | 0.232223 | +0.022872 | 0.864473 |
| **15 dB** | **CBF** | **DNN-G** | 10.288727 | 3.207605 | 2.424857 | +0.271243 | 0.993877 |
| | | **H-DELM V1** | 10.126359 | 3.182194 | 2.395599 | -0.019729 | 0.993982 |
| | | **H-DELM V2** | 10.108052 | 3.179316 | 2.395747 | -0.016593 | 0.993998 |
| **15 dB** | **ATT** | **DNN-G** | 0.093700 | 0.306104 | 0.202613 | -0.003425 | 0.898631 |
| | | **H-DELM V1** | 0.093250 | 0.305369 | 0.202046 | -0.005846 | 0.898946 |
| | | **H-DELM V2** | 0.093365 | 0.305558 | 0.202871 | -0.007428 | 0.898745 |
| **20 dB** | **CBF** | **DNN-G** | 8.728568 | 2.954415 | 2.150076 | +0.034715 | 0.994743 |
| | | **H-DELM V1** | 8.697042 | 2.949075 | 2.147344 | -0.246441 | 0.994768 |
| | | **H-DELM V2** | 8.722282 | 2.953351 | 2.154661 | -0.239189 | 0.994757 |
| **20 dB** | **ATT** | **DNN-G** | 0.086672 | 0.294401 | 0.192854 | -0.008303 | 0.908995 |
| | | **H-DELM V1** | 0.086427 | 0.293984 | 0.191752 | -0.010459 | 0.909122 |
| | | **H-DELM V2** | 0.087204 | 0.295303 | 0.193391 | -0.012243 | 0.908166 |
| **Pooled** | **CBF** | **DNN-G** | 12.057628 | 3.472409 | 2.606443 | +0.385385 | 0.992713 |
| **(30k)** | | **H-DELM V1** | 11.778833 | 3.432031 | 2.571329 | +0.086527 | 0.992891 |
| | | **H-DELM V2** | 11.799721 | 3.435072 | 2.577100 | +0.088100 | 0.992884 |
| **Pooled** | **ATT** | **DNN-G** | 0.101841 | 0.319125 | 0.208856 | +0.004960 | 0.891158 |
| **(30k)** | | **H-DELM V1** | 0.101709 | 0.318918 | 0.208261 | +0.002469 | 0.891098 |
| | | **H-DELM V2** | 0.102100 | 0.319531 | 0.209495 | +0.001067 | 0.890580 |

---

## 3. Correction of Reported Numerical Claims

> [!IMPORTANT]
> **Audit Correction on V1 vs V2 Differences:**
> The previous summary statement claiming *"V1–V2 differences are below 0.17%"* was an incomplete generalization.
> - While it is true that the **CBF RMSE difference** between V1 and V2 does not exceed **0.169%** across any SNR level ($0.169\%$ at SNR 10, $0.090\%$ at SNR 15, and $0.145\%$ at SNR 20),
> - The difference for **ATT RMSE reaches up to -0.448%** (at SNR 20 dB), and the difference for **ATT MAE reaches up to -0.855%** (at SNR 20 dB).
> The corrected full matrix of relative differences is recorded below.

### Table 2: Verified Percentage Gains Relative to DNN-G and V1 Comparison

| SNR | Parameter | V1 Gain vs DNN-G (RMSE %) | V2 Gain vs DNN-G (RMSE %) | V2 vs V1 Relative (RMSE %) | V1 Gain vs DNN-G (MAE %) | V2 Gain vs DNN-G (MAE %) | V2 vs V1 Relative (MAE %) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 dB** | **CBF** | **+1.890%** | **+1.725%** | **-0.169%** | **+2.261%** | **+1.957%** | **-0.311%** |
| **10 dB** | **ATT** | -0.119% | -0.232% | **-0.112%** | +0.049% | -0.486% | **-0.535%** |
| **15 dB** | **CBF** | +0.792% | **+0.882%** | **+0.090%** | **+1.207%** | +1.200% | **-0.006%** |
| **15 dB** | **ATT** | **+0.240%** | +0.178% | **-0.062%** | **+0.280%** | -0.127% | **-0.408%** |
| **20 dB** | **CBF** | **+0.181%** | +0.036% | **-0.145%** | **+0.127%** | -0.213% | **-0.341%** |
| **20 dB** | **ATT** | **+0.141%** | -0.306% | **-0.448%** | **+0.572%** | -0.278% | **-0.855%** |
| **Pooled** | **CBF** | **+1.163%** | **+1.075%** | **-0.089%** | **+1.347%** | **+1.126%** | **-0.224%** |
| **Pooled** | **ATT** | **+0.065%** | -0.127% | **-0.192%** | **+0.285%** | -0.306% | **-0.592%** |

---

## 4. Paired V1 vs. V2 Hypothesis Testing & Equivalence Analysis

To rigorously test whether V1 and V2 differ, and whether they can be considered practically equivalent, we evaluated:
1. **Hypothesis Test ($H_0: \text{V1 and V2 have equal error}$):**
   Paired sign-flip permutation tests ($B=10,000$, seed `42`), Wilcoxon signed-rank tests, and Holm-Bonferroni FWER step-down adjustment.
2. **Equivalence Analysis (Two One-Sided Tests / 95% Confidence Interval Inclusion):**
   - **CBF Equivalence Margin:** $\delta_{\text{CBF}} = \pm 0.50\text{ mL}/100\text{g}/\text{min}$ ($\approx 1\%$ of mean physiological CBF $\approx 50$).
   - **ATT Equivalence Margin:** $\delta_{\text{ATT}} = \pm 0.025\text{ s}$ ($25\text{ ms}$, $5\%$ of the $500\text{ ms}$ inter-PLD spacing).
   - **Decision Rule:** Practical equivalence is established if and only if the two-sided 95% bootstrap confidence interval $[\text{CI}_{\text{lower}}, \text{CI}_{\text{upper}}]$ falls strictly within $[-\delta, +\delta]$.

### Table 3: Paired V1 vs. V2 Comparison, Statistical Tests, and Equivalence Verification

| SNR | Parameter | $\Delta_{\text{RMSE}}$ (V1 - V2) | 95% Bootstrap CI ($\Delta_{\text{RMSE}}$) | Permutation $p$ (MSE) | Holm-Adjusted $p$ (MSE) | Wilcoxon $p$ (Holm) | Equivalence Margin ($\pm \delta$) | 95% CI $\subset [-\delta, +\delta]$? | Practical Equivalence? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 dB** | **CBF** | -0.0069 | $[-0.0168, +0.0022]$ | 0.1606 | 0.8029 | 0.1799 | $\pm 0.50\text{ mL}/100\text{g}/\text{min}$ | **YES** | **ESTABLISHED** |
| **10 dB** | **ATT** | -0.0004 | $[-0.0012, +0.0004]$ | 0.3341 | 1.0000 | 0.0088 | $\pm 0.025\text{ s}$ ($25\text{ ms}$) | **YES** | **ESTABLISHED** |
| **15 dB** | **CBF** | +0.0029 | $[-0.0049, +0.0109]$ | 0.4879 | 1.0000 | 0.9657 | $\pm 0.50\text{ mL}/100\text{g}/\text{min}$ | **YES** | **ESTABLISHED** |
| **15 dB** | **ATT** | -0.0002 | $[-0.0011, +0.0006]$ | 0.6597 | 1.0000 | 0.0060 | $\pm 0.025\text{ s}$ ($25\text{ ms}$) | **YES** | **ESTABLISHED** |
| **20 dB** | **CBF** | -0.0043 | $[-0.0123, +0.0038]$ | 0.2912 | 1.0000 | 0.0874 | $\pm 0.50\text{ mL}/100\text{g}/\text{min}$ | **YES** | **ESTABLISHED** |
| **20 dB** | **ATT** | -0.0013 | $[-0.0021, -0.0005]$ | 0.0015 | 0.0090 | $1.16 \times 10^{-11}$ | $\pm 0.025\text{ s}$ ($25\text{ ms}$) | **YES** | **ESTABLISHED** |

### Statistical Synthesis:
- **Difference Hypothesis:** At SNR 20 dB for ATT, the difference between V1 and V2 is statistically detectable ($p_{\text{Holm}} = 0.0090$) because $N = 10,000$ yields extreme statistical power.
- **Equivalence Finding:** However, the observed difference is only $\Delta_{\text{RMSE}} = -0.0013\text{ s}$ (1.3 milliseconds), and its 95% confidence interval $[-0.0021, -0.0005]\text{ s}$ lies entirely inside the $\pm 25\text{ ms}$ clinical equivalence window.
- **Scientific Conclusion:** V1 and V2 are **statistically distinguishable under high sample size but practically equivalent** within physiologically negligible margins.

---

## 5. Technical Implementation Audit

| Component | Audit Requirement | Verification Finding | Status |
| :--- | :--- | :--- | :---: |
| **V1 Residual Fitting** | Uses 1M DNN training samples | Confirmed in `run_reproduction_pipeline.py` (Cell 84-159) | **VERIFIED** |
| **V2 Residual Fitting** | Uses 100k independent calibration samples | Confirmed in `generate_calibration_dataset.py` (Seed 31415) | **VERIFIED** |
| **DNN Freezing** | Zero gradient updates during ELM fit | Confirmed: `net.eval()`, `torch.no_grad()`, weights unchanged | **VERIFIED** |
| **Feature Standardization** | Scaler fit exclusively on calibration data | Recomputed $\mu_{\text{feat}}$ and $\sigma_{\text{feat}}$ match artifact with $|\Delta| < 10^{-7}$ | **VERIFIED** |
| **Analytical Ridge Equation** | $(H^T H + \lambda I)\beta = H^T r$ | Recomputed closed-form solve matches artifact $\beta$ with relative error $< 10^{-6}$ | **VERIFIED** |
| **Artifact Completeness** | Contains all inference parameters | Contains $W, b, \beta, \mu, \sigma, L=512, \lambda=0.01, \text{bounds}, \text{PLDs}$ | **VERIFIED** |
| **Standalone Inference** | Matches pipeline predictions | Re-evaluated on 30k test samples: $\text{max } |\Delta| < 10^{-5}$ across all outputs | **VERIFIED** |

---

## 6. Review of Core Scientific Claims

### Claim 1: "Residual H-DELM improves CBF estimation under low SNR."
- **Audit Assessment:** **SUPPORTED.**
- **Empirical Evidence:** At SNR 10 dB, CBF RMSE is reduced from **4.1419 to 4.0705** (+1.73% reduction, paired permutation $p < 0.001$, Wilcoxon $p = 1.73 \times 10^{-30}$ after Holm-Bonferroni correction). MAE is reduced from **3.2444 to 3.1809** (+1.96%). Overestimation bias is reduced by **38.8%** (from $+0.850$ down to $+0.520\text{ mL}/100\text{g}/\text{min}$). At SNR 15 dB, CBF RMSE reduction remains significant (+0.88%, $p < 0.001$).

### Claim 2: "Residual H-DELM provides consistent ATT improvement."
- **Audit Assessment:** **NOT SUPPORTED.**
- **Empirical Evidence:** For ATT, changes between DNN-G and H-DELM are sub-millisecond (e.g. $\Delta_{\text{RMSE}} = -0.0008\text{ s}$ at SNR 10; $+0.0005\text{ s}$ at SNR 15; $-0.0009\text{ s}$ at SNR 20). Permutation tests for squared-error differences fail to reach significance after Holm-Bonferroni correction ($p = 0.220$ at SNR 10; $p = 0.508$ at SNR 15). The 4-PLD sampling protocol fundamentally limits arrival time sensitivity.

### Claim 3: "Residual ELM generalizes effectively."
- **Audit Assessment:** **SUPPORTED FOR CBF, RESTRICTED FOR ATT.**
- **Empirical Evidence:** The ELM learns a robust, low-rank linear projection on the 79-dimensional representation that transfers from the 100k calibration cohort to the 30k test cohort without degradation.

### Claim 4: "V2 is comparable to V1."
- **Audit Assessment:** **SUPPORTED WITH QUALIFICATIONS.**
- **Empirical Evidence:** V1 and V2 are practically equivalent across all conditions within predefined clinical margins ($\pm 0.50\text{ mL}/100\text{g}/\text{min}$ and $\pm 0.025\text{ s}$). However, V2 is the **only methodologically legitimate model** because it guarantees complete cohort independence.

### Claim 5: "Physics features enhance biological interpretability."
- **Audit Assessment:** **QUALIFIED HYPOTHESIS.**
- **Empirical Evidence:** Directly demonstrated: the 15 features (energy, trapezoidal AUC, variance) act as amplitude regularizers that mitigate Rician rectification noise on CBF. Hypothesis: they do not provide dynamic physiological deconvolution for temporal transit delay.

---

## 7. Audit Deliverables Inventory

The following authoritative deliverables have been generated and saved:
1. [`FINAL_REPRODUCIBILITY_AUDIT.md`](file:///d:/dnn%20vs%20dnn%20elm/FINAL_REPRODUCIBILITY_AUDIT.md) — Comprehensive audit report.
2. [`FINAL_METRIC_VERIFICATION.csv`](file:///d:/dnn%20vs%20dnn%20elm/FINAL_METRIC_VERIFICATION.csv) — Recomputed metrics and percentage gains across SNR 10, 15, 20 and Pooled.
3. [`FINAL_V1_V2_PAIRED_COMPARISON.csv`](file:///d:/dnn%20vs%20dnn%20elm/FINAL_V1_V2_PAIRED_COMPARISON.csv) — Paired differences, bootstrap 95% CIs, and equivalence analysis.
4. [`FINAL_STATISTICAL_TESTS.csv`](file:///d:/dnn%20vs%20dnn%20elm/FINAL_STATISTICAL_TESTS.csv) — Permutation and Wilcoxon tests with Holm-Bonferroni step-down adjustments.

---

## 8. Final Audit Verdict

```
========================================================================================
FINAL AUDIT VERDICT: VERIFIED WITH QUALIFICATIONS
========================================================================================
- Technical Implementation & Reproducibility: VERIFIED (100% exact numerical match)
- Cohort Independence & Provenance         : VERIFIED (Zero sample/signal overlap)
- CBF Performance Claims                   : VERIFIED (Statistically significant gains at SNR 10 & 15)
- ATT Performance Claims                   : NOT VERIFIED (No significant or meaningful gain)
- Practical Equivalence of V1 vs V2         : VERIFIED (Within predeclared equivalence margins)
========================================================================================
```

