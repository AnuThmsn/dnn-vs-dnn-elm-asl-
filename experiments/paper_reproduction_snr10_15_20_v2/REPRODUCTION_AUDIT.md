# REPRODUCTION AUDIT REPORT (VERSION 2)

**Study Title:** *A Residual Hybrid DNN–ELM Model for CBF and ATT Estimation in Multi-Delay ASL MRI*  
**Audited Directory:** `experiments/paper_reproduction_snr10_15_20_v2`  
**Evaluation Scope:** Scientific Protocol Correction of Residual H-DELM (Independent Calibration vs Training Fit)  
**Date of Audit:** November 2026  

---

## 1. Discrepancy Identified in V1 Protocol

In the initial reproduction experiment (V1, located in `experiments/paper_reproduction_snr10_15_20/`), the residual ELM was trained directly on the **same 1,000,000 observations** used to train the baseline DNN backbones.

### Methodological Concern:
- Training the residual corrector on training-set residuals means the ELM observes training residuals:
  $$r_{\text{train}} = y_{\text{train}} - \hat{y}_{\text{DNN}}(x_{\text{train}})$$
- Because the DNN has already minimized empirical loss on its training cohort, training-set residuals are partially fitted and may contain idiosyncratic memorization artifacts rather than true out-of-distribution generalization residuals.
- In contrast, the theoretical formulation described in the paper envisions the residual model acting as an independent calibrator/regularizer fitted on independently simulated pairs:
  $$r_{\text{cal}} = y_{\text{cal}} - \hat{y}_{\text{DNN}}(x_{\text{cal}})$$
  where $(x_{\text{cal}}, y_{\text{cal}})$ is held out from DNN backpropagation.

---

## 2. Protocol Modifications in V2

| Parameter / Step | V1 Protocol (Baseline Reproduction) | V2 Protocol (Corrected Pipeline) | Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **Residual Training Source** | 1,000,000 DNN training signals | **100,000 Independent Calibration Signals** | Eliminates residual memorization from DNN training set |
| **Calibration Seed** | Seed 42 (shared with DNN train) | **Seed 31415 (Held-out PRNG stream)** | Guarantees zero data leakage or sample overlap |
| **Feature Standardization** | Fit on 1M DNN training samples | **Fit strictly on 100k calibration samples** | Standardizes 79-dim representations on independent cohort |
| **DNN Backbones** | Trained on 1M signals (4x64, ELU) | **Frozen V1 Checkpoints Reused Directly** | Perfectly isolates the effect of the calibration change |
| **Test Dataset** | 30k independent samples (`seed 2026`) | **Exact same 30k test cohort (`independent_test_30k.npz`)** | Strict paired comparison on identical test observations |
| **Hyperparameter Grid** | Fixed $\lambda = 0.01$ | **Sensitivity sweep on grid $[10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0]$** | Verifies numerical stability without touching test set |
| **Statistical Hypothesis Testing** | Paired bootstrap & Wilcoxon | **Paired permutation test (10,000 sign-flips) + Holm-Bonferroni** | Rigorously controls Family-Wise Error Rate (FWER) |
| **Model Persistence** | None (in-memory only) | **Complete standalone `.npz` ELM artifacts** | Enables standalone zero-retraining clinical inference |

---

## 3. Provenance of Default Hyperparameters

Forensic audit of `dnn-elm-noise (7).ipynb` revealed the exact origin of default hyperparameters:
- **Hidden Neuron Count ($L=512$):** In Cell 18/19 of the notebook, a hidden-neuron sweep evaluated $\{128, 256, 512, 1024\}$ neurons and selected 512 as balancing model capacity against numerical conditioning.
- **Ridge Regularization ($\lambda = 0.01$):** In Cell 15 of the notebook, `regularization=1e-2` was hardcoded as the default Tikhonov regularization scalar.
- **Random Projection Seed ($42$):** Standard PRNG seed for uniform random projection initialization $W \sim U[-1/\sqrt{79}, 1/\sqrt{79}]$.

---

## 4. Verification Checkpoints

1. **Exact Baseline Identity:** Confirmed that DNN-G predictions for Group A are identical between V1 and V2 ($|\Delta| = 0.0$).
2. **Standalone Inference Validation:** Verified that `StandaloneASLEstimator` reproduces pipeline predictions within machine precision ($\text{atol} = 10^{-5}$).
3. **Physiological Bound Enforcement:** Verified zero NaNs, zero infinities, and 100% adherence to physical limits:
   - $\text{CBF} \in [0.0, 100.0]\text{ mL}/100\text{g}/\text{min}$
   - $\text{ATT} \in [0.5, 3.0]\text{ s}$

