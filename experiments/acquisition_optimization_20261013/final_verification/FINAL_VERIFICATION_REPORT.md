# Final Forensic Verification Report: Optimized ASL Acquisition

**Verification Directory:** `experiments/acquisition_optimization_20261013/final_verification/`  
**Date:** October 13, 2026  
**Audited Target Report:** `experiments/acquisition_optimization_20261013/ACQUISITION_OPTIMIZATION_REPORT.md`  
**Reference Commit:** `fbf875e`  
**Verification Protocol:** Zero retraining, zero dataset regeneration, strictly independent recalculation from saved artifacts on Drive D.  

---

## 1. Executive Summary & Verdict

This audit conducted an independent, forensic verification of the results, training histories, physical identifiability claims, and test metrics reported for the Optimized ASL Acquisition campaign ($[0.40, 1.10, 2.00, 3.00]\text{ s}$ vs. Baseline $[1.525, 2.025, 2.525, 3.025]\text{ s}$).

### Overall Verification Verdict: **FULLY VERIFIED**
Every quantitative claim in `ACQUISITION_OPTIMIZATION_REPORT.md` is reproducible from the saved raw prediction arrays within numerical precision ($< 10^{-15}$). The training curves are healthy, monotonic, and match the saved checkpoints. The physical identifiability explanation is verified mathematically and by central finite difference derivatives.

### Key Verified Metrics on Locked Test Set ($N = 100,000$)

| Metric | Baseline A (DNN-G) | Baseline Feature ELM | Optimized 4-PLD DNN | Optimized 4-PLD Feature ELM | Recomputed Gain (%) | Audit Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CBF RMSE** | 3.7266 | 3.6828 | 3.1020 | **3.0846** | **+17.23%** | **VERIFIED** |
| **CBF MAE** | 2.5323 | 2.4232 | 1.7129 | **1.6249** | **+35.83%** | **VERIFIED** |
| **CBF Lin's CCC** | 0.99157 | 0.99179 | 0.99419 | **0.99425** | $\Delta = +0.00268$ | **VERIFIED** |
| **ATT RMSE (s)** | 0.30580 | 0.30487 | **0.18206** | 0.18216 | **+40.47%** | **VERIFIED** |
| **ATT MAE (s)** | 0.19352 | 0.19039 | 0.07311 | **0.07097** | **+63.33%** | **VERIFIED** |
| **ATT Lin's CCC** | 0.90092 | 0.90161 | **0.96716** | 0.96698 | $\Delta = +0.06624$ | **VERIFIED** |
| **Early ATT RMSE (s)** | 0.46643 | 0.46933 | 0.22610 | **0.21710** | **+53.45%** | **VERIFIED** |
| **Early ATT MAE (s)** | 0.36910 | 0.37161 | 0.07086 | **0.06763** | **+81.68%** | **VERIFIED** |
| **Early ATT Bias (s)** | +0.36905 | +0.37155 | +0.05240 | **+0.04931** | **86.6% eliminated** | **VERIFIED** |
| **Early ATT CCC** | 0.00469 | 0.00503 | 0.43453 | **0.45691** | **97-fold increase** | **VERIFIED** |

---

## 2. Phase 1 Audit: Actual Training Curves & Checkpoint Integrity

The training history was extracted directly from `checkpoints/training_history_opt.json` and compared against Baseline A in `baseline_A/training_history.json`:

### 2.1 Optimized ATTnet Training History
- **Total Epochs Trained:** 30
- **Training Time:** $895.3\text{ seconds}$
- **Initial Loss (Epoch 1):** $\text{Train} = 0.084866, \text{Val} = 0.069120$
- **Best Validation Epoch:** **Epoch 20** ($\text{Val Loss} = \mathbf{0.060993}$, $\text{Train Loss} = 0.064780$)
- **Stopping Behavior:** No improvement from Epoch 21 through 30; early stopping triggered exactly at Epoch 30 (patience $= 10$).
- **Saved Checkpoint:** `checkpoints/ATTnet_opt_4PLD.pth` correctly restored the state dict from Epoch 20.
- **Diagnostic Plot:** [training_curves_att.png](file:///d:/dnn%20vs%20dnn%20elm/experiments/acquisition_optimization_20261013/final_verification/training_curves_att.png)

### 2.2 Optimized CBFnet Training History
- **Total Epochs Trained:** 20
- **Training Time:** $596.5\text{ seconds}$
- **Initial Loss (Epoch 1):** $\text{Train} = 0.021087, \text{Val} = 0.013931$
- **Best Validation Epoch:** **Epoch 10** ($\text{Val Loss} = \mathbf{0.011518}$, $\text{Train Loss} = 0.011818$)
- **Stopping Behavior:** No improvement from Epoch 11 through 20; early stopping triggered at Epoch 20 (patience $= 10$).
- **Saved Checkpoint:** `checkpoints/CBFnet_opt_4PLD.pth` correctly restored the state dict from Epoch 10.
- **Diagnostic Plot:** [training_curves_cbf.png](file:///d:/dnn%20vs%20dnn%20elm/experiments/acquisition_optimization_20261013/final_verification/training_curves_cbf.png)

### 2.3 Curve Analysis & Overfitting Audit
1. **Convergence Stability:** Both networks exhibit smooth, asymptotic loss decay without spikes, oscillations, or gradient divergence.
2. **Generalization Gap:** The training and validation loss curves track each other tightly (gap $< 0.004$ standardized units), showing no signs of memorization or variance overfitting.
3. **Loss Scale Comparison:** Because both models minimize standardized MSE ($\text{MSE} / \sigma_y^2$), their validation losses are directly comparable to Baseline A:
   - **ATT Validation Loss:** Dropped from **$0.177601$ (Baseline)** to **$0.060993$ (Optimized)**, reflecting an intrinsic error variance reduction of **$65.7\%$**.
   - **CBF Validation Loss:** Dropped from **$0.016791$ (Baseline)** to **$0.011518$ (Optimized)**, reflecting a **$31.4\%$** reduction in validation MSE.
4. **Feature ELM Note:** The Feature ELM uses an analytical closed-form Moore-Penrose ridge solver on penultimate latents. As verified, no iterative epochs were fabricated for ELM fitting.
5. **Comparison Plot:** [training_curves_comparison.png](file:///d:/dnn%20vs%20dnn%20elm/experiments/acquisition_optimization_20261013/final_verification/training_curves_comparison.png)

---

## 3. Phase 2 Audit: Independent Recalculation of Test Metrics

Predictions were loaded from `checkpoints/optimized_test_predictions.npz` and re-evaluated from scratch in `final_verification/verify_all.py`:

### 3.1 Data Integrity Checks
- `Y_cbf_test`: shape $(100000,)$, range $[0.0022, 99.9990]$, zero NaNs, zero Infs (**PASS**).
- `Y_att_test`: shape $(100000,)$, range $[0.5000, 3.0000]$, zero NaNs, zero Infs (**PASS**).
- `p_att_base_dnn`: shape $(100000,)$, range $[\mathbf{0.9314}, 3.0000]$.  
  *Crucial Finding:* The baseline DNN **never predicted any ATT value below $0.9314\text{ s}$**, proving complete blindness to the early-arrival regime $[0.5, 0.9]\text{ s}$.
- `p_att_opt_dnn`: shape $(100000,)$, range $[\mathbf{0.5000}, 3.0000]$.  
  *Crucial Finding:* The optimized DNN spans the entire physiological range down to the lower boundary of $0.5000\text{ s}$.

### 3.2 Numerical Tolerance Cross-Check
Comparing independently recomputed metrics against the reported values in `tables/optimized_vs_baseline_comparison.csv`:
- Maximum ATT RMSE difference: $5.55 \times 10^{-17}$
- Maximum CBF RMSE difference: $4.44 \times 10^{-16}$
- Maximum MAE difference: $0.00 \times 10^{-17}$
- **Verdict:** Absolute exact numerical match across all models and metrics. Saved in `final_verification/recomputed_metrics.csv`.

---

## 4. Phase 3 Audit: Fairness of the Experimental Comparison

1. **Ground-Truth Parameter Matching:**  
   The test ground-truth arrays $Y_{\text{test}}$ from `experiments/cache/Y_test.npy` and `checkpoints/optimized_test_predictions.npz` were subtracted directly.  
   $\max |Y_{\text{orig}} - Y_{\text{opt}}| = 0.000000$ (**Exactly Identical**).
2. **Noise Realization Matching:**  
   Noise generation used the exact same test seed (`seed = 7`) and SNR assignments ($100,000$ elements matched across 12 levels).
3. **Channel-Specific Input Validity:**  
   - Baseline models evaluated signals generated with $\text{PLDs} = [1.525, 2.025, 2.525, 3.025]\text{ s}$ ($\text{mean} = 491.11$).
   - Optimized models evaluated signals generated with $\text{PLDs} = [0.40, 1.10, 2.00, 3.00]\text{ s}$ ($\text{mean} = 526.08$).
   - No cross-PLD contamination or silent channel swapping occurred.
4. **Leakage & Normalization Audit:**  
   - Normalization statistics for the optimized network (`normalization_stats_opt.npz`) were fitted solely on $X_{\text{train}}^{\text{opt}}$ ($N=1,000,000$).
   - The test set was strictly locked and excluded from model selection and early stopping.
5. **Scientific Attribution:**  
   The observed gains measure the combined system performance of **optimal physical acquisition timing coupled with a matched estimator**, rather than an estimator architecture improvement on fixed data.

---

## 5. Phase 4 Audit: Forward-Model Identifiability Claim

In `final_verification/identifiability_audit.csv`, the analytical Jacobian was tested against central finite difference derivatives across 11 ATT evaluation points ($0.50\text{ s} \to 3.00\text{ s}$):

### 5.1 Derivative Accuracy
$$\max |J_{\text{analytical}} - J_{\text{numerical}}| < 9.9 \times 10^{-9}$$
The analytical Jacobian is mathematically exact.

### 5.2 Collinearity Audit

| ATT (s) | Baseline Cosine Collinearity $|\cos\theta|$ | Baseline $\log_{10} \kappa(J)$ | Baseline $\sigma_2$ | Optimized Cosine Collinearity $|\cos\theta|$ | Optimized $\log_{10} \kappa(J)$ | Optimized $\sigma_2$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.50** | **1.000000** | **16.0** | $2.5 \times 10^{-15}$ | **0.7008** | **1.71** | **14.82** |
| **0.60** | **1.000000** | **16.0** | $2.9 \times 10^{-15}$ | **0.6511** | **1.68** | **15.11** |
| **0.75** | **1.000000** | **16.0** | $3.9 \times 10^{-15}$ | **0.5649** | **1.64** | **15.34** |
| **0.90** | **1.000000** | **16.0** | $3.8 \times 10^{-15}$ | **0.4651** | **1.60** | **15.48** |
| **1.00** | **1.000000** | **16.0** | $3.8 \times 10^{-15}$ | **0.3917** | **1.58** | **15.52** |
| **1.25** | **1.000000** | **16.0** | $2.6 \times 10^{-15}$ | **0.8300** | **1.93** | **7.91** |
| **1.50** | **1.000000** | **16.0** | $1.8 \times 10^{-15}$ | **0.7201** | **1.86** | **9.05** |
| **1.75** | 0.4630 | 1.56 | 10.44 | 0.5469 | 1.80 | 9.27 |
| **2.00** | 0.2814 | 1.50 | 10.66 | 0.3027 | 1.75 | 9.42 |
| **2.50** | 0.4963 | 1.67 | 7.48 | 0.6657 | 1.89 | 5.18 |
| **3.00** | 0.5886 | 1.83 | 4.81 | 0.3105 | 1.63 | 6.21 |

### 5.3 Audit Conclusions
1. For every $\text{ATT} \le 1.50\text{ s}$, Baseline A has **exact collinearity** ($|\cos\theta| = 1.000000$) and rank-1 collapse ($\sigma_2 \approx 0$).
2. The optimized acquisition drops condition numbers from $10^{16}$ to $\le 85$ across all ATT values.
3. At the extreme early boundary $\text{ATT} = 0.50\text{ s}$, the optimized protocol maintains $\sigma_2 = 14.82$ and $\kappa(J) = 50.8$, ensuring stable identifiability across the entire parameter space.

---

## 6. Phase 5 Audit: Early-Arrival and Interval Verification

Recomputed metrics across the 5 predefined intervals in `final_verification/interval_recomputation.csv`:

### 6.1 Cohort Partitioning Audit
- $[0.5, 1.0)\text{ s}$: $19,728$
- $[1.0, 1.5)\text{ s}$: $20,281$
- $[1.5, 2.0)\text{ s}$: $19,836$
- $[2.0, 2.5)\text{ s}$: $20,275$
- $[2.5, 3.0]\text{ s}$: $19,880$
- **Total Sum:** **$100,000$** (**Exact match with full test cohort**).

### 6.2 Early-Arrival Window $[0.5, 1.0)\text{ s}$ Recomputation ($N=19,728$)
- **Baseline A (DNN-G):** $\text{RMSE} = 0.466430\text{ s}, \text{MAE} = 0.369102\text{ s}, \text{Bias} = +0.369054\text{ s}, \text{CCC} = 0.004686$.
- **Optimized 4-PLD DNN:** $\text{RMSE} = 0.226101\text{ s}, \text{MAE} = 0.070855\text{ s}, \text{Bias} = +0.052401\text{ s}, \text{CCC} = 0.434529$.
- **Optimized 4-PLD Feature ELM:** $\text{RMSE} = 0.217101\text{ s}, \text{MAE} = 0.067631\text{ s}, \text{Bias} = +0.049308\text{ s}, \text{CCC} = 0.456914$.
- **Verified Gained Reductions:**
  - ATT RMSE reduced by **$53.45\%$** ($0.4664 \to 0.2171\text{ s}$).
  - ATT MAE reduced by **$81.68\%$** ($0.3691 \to 0.0676\text{ s}$).
  - Bias reduced by **$86.64\%$** ($+0.3691 \to +0.0493\text{ s}$).
  - CCC increased by **$96.4\times$** ($0.0047 \to 0.4569$).
- Diagnostic figure generated: [recomputed_interval_profile.png](file:///d:/dnn%20vs%20dnn%20elm/experiments/acquisition_optimization_20261013/final_verification/recomputed_interval_profile.png).

---

## 7. Phase 6 Audit: Statistical Validation & Uncertainty Bounds

Recalculated in `final_verification/statistical_recomputation.csv` on all $100,000$ matched observations with 1,000 bootstrap resamples:

### 7.1 Statistical Summary Table

| Comparison | Target | Difference Metric | Point Improvement ($\Delta$) | 95% Bootstrap CI | Paired t-test Statistic ($t$) | $p$-value Bound |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Optimized DNN vs. Baseline A** | CBF | RMSE | $+0.6245\text{ mL}/(100\text{g}\cdot\text{min})$ | $[0.5827, 0.6639]$ | $t = 30.12$ (on MSE) | $p = 3.84 \times 10^{-198}$ |
| **Optimized DNN vs. Baseline A** | CBF | MAE | $+0.8194\text{ mL}/(100\text{g}\cdot\text{min})$ | $[0.7995, 0.8379]$ | $t = 53.71$ (on MAE) | $p < 10^{-300}$ |
| **Optimized DNN vs. Baseline A** | ATT | RMSE | $+0.1237\text{ s}$ ($123.7\text{ ms}$) | $[0.12108, 0.12673]\text{ s}$ | $t = 61.42$ (on MSE) | $p < 10^{-300}$ |
| **Optimized DNN vs. Baseline A** | ATT | MAE | $+0.1204\text{ s}$ ($120.4\text{ ms}$) | $[0.11909, 0.12184]\text{ s}$ | $t = 118.24$ (on MAE) | $p < 10^{-300}$ |
| **Optimized ELM vs. Baseline A** | CBF | RMSE | $+0.6420\text{ mL}/(100\text{g}\cdot\text{min})$ | $[0.5980, 0.6875]$ | $t = 30.78$ (on MSE) | $p = 2.57 \times 10^{-206}$ |
| **Optimized ELM vs. Baseline A** | CBF | MAE | $+0.9074\text{ mL}/(100\text{g}\cdot\text{min})$ | $[0.8877, 0.9286]$ | $t = 61.85$ (on MAE) | $p < 10^{-300}$ |
| **Optimized ELM vs. Baseline A** | ATT | RMSE | $+0.1236\text{ s}$ ($123.6\text{ ms}$) | $[0.12075, 0.12654]\text{ s}$ | $t = 61.35$ (on MSE) | $p < 10^{-300}$ |
| **Optimized ELM vs. Baseline A** | ATT | MAE | $+0.1226\text{ s}$ ($122.6\text{ ms}$) | $[0.12124, 0.12384]\text{ s}$ | $t = 121.06$ (on MAE) | $p < 10^{-300}$ |

*Reporting Compliance Note:* Floating-point underflows are explicitly bounded as $p < 10^{-300}$ rather than incorrectly reported as $0.0$.

---

## 8. Artifact and Provenance Verification

| Artifact | File Path | File Size | SHA-256 Checksum | Status |
|:---|:---|:---:|:---|:---:|
| **Optimized ATTnet Model** | `checkpoints/ATTnet_opt_4PLD.pth` | $55.3\text{ KB}$ | `d4e518bc25b29db52c8b7c7b8e11a3df3f48d80f5d0f624d78ad7fc38e2119ec` | **VERIFIED** |
| **Optimized CBFnet Model** | `checkpoints/CBFnet_opt_4PLD.pth` | $272.8\text{ KB}$ | `b1e06f9d342be33550993510c4980a3a41e976db57fc3e88efb8b2fa808722c1` | **VERIFIED** |
| **Optimized Predictions** | `checkpoints/optimized_test_predictions.npz` | $5.34\text{ MB}$ | `53fe1e6a96f131a4c9b9877b0df050519ea81bbf5ef2c2c040d9b9961603d3c7` | **VERIFIED** |
| **Training History** | `checkpoints/training_history_opt.json` | $7.33\text{ KB}$ | `7335d886981121d58bc4e3b1c69db3fae1feaa806fcf88b56f2f3072f963ca5b` | **VERIFIED** |
| **Normalization Stats** | `checkpoints/normalization_stats_opt.npz` | $1.56\text{ KB}$ | `39f11654e99fcaeecc8eb25bf9c9a0937a077a28e81d77a06f357ff8e83be62f` | **VERIFIED** |

---

## 9. Comprehensive Claim Classification Table

| # | Scientific Claim | Evaluated Evidence | Classification | Rationale |
|:---:|:---|:---|:---:|:---|
| 1 | **Baseline early ATT is rank-1 unidentifiable** | Analytical & numerical Jacobians | **VERIFIED** | Cosine collinearity is identically $1.000000$ for all $\text{ATT} \le 1.525\text{ s}$; condition number $\kappa \ge 10^{16}$. |
| 2 | **Baseline DNN cannot estimate early ATT** | Saved predictions on $[0.5, 1.0)\text{ s}$ | **VERIFIED** | Range of $\hat{\delta} \in [0.9314, 3.0000]$; never predicts below $0.93\text{ s}$; invariant bias $+0.369\text{ s}$, $\text{CCC} = 0.0047$. |
| 3 | **$[0.40, 1.10, 2.00, 3.00]\text{ s}$ restores identifiability** | Condition number & singular values | **VERIFIED** | $\sigma_2$ jumps from $0 \to 14.8 - 15.5$, $\kappa(J) \le 85$, collinearity drops to $0.39 - 0.70$. |
| 4 | **Global ATT RMSE improves by $\sim 40\%$** | Recalculated on $100,000$ test cases | **VERIFIED** | $0.30580\text{ s} \to 0.18206\text{ s}$ (**$+40.47\%$ reduction**, $p < 10^{-300}$). |
| 5 | **Global ATT MAE improves by $\sim 63\%$** | Recalculated on $100,000$ test cases | **VERIFIED** | $0.19352\text{ s} \to 0.07097\text{ s}$ (**$+63.33\%$ reduction**, $p < 10^{-300}$). |
| 6 | **Early ATT RMSE improves by $> 50\%$** | Recalculated on $19,728$ early cases | **VERIFIED** | $0.4664\text{ s} \to 0.2171\text{ s}$ (**$+53.45\%$ reduction**). |
| 7 | **Early ATT MAE improves by $> 80\%$** | Recalculated on $19,728$ early cases | **VERIFIED** | $0.3691\text{ s} \to 0.0676\text{ s}$ (**$+81.68\%$ reduction**). |
| 8 | **Early ATT bias is reduced by $> 85\%$** | Recalculated on $19,728$ early cases | **VERIFIED** | Bias drops from $+0.3691\text{ s} \to +0.0493\text{ s}$ (**$86.6\%$ reduction**). |
| 9 | **Global CBF RMSE improves by $\sim 17\%$** | Recalculated on $100,000$ test cases | **VERIFIED** | $3.7266 \to 3.0846\text{ mL}/(100\text{g}\cdot\text{min})$ (**$+17.23\%$ reduction**, $p = 2.57 \times 10^{-206}$). |
| 10 | **Global CBF MAE improves by $\sim 35\%$** | Recalculated on $100,000$ test cases | **VERIFIED** | $2.5323 \to 1.6249\text{ mL}/(100\text{g}\cdot\text{min})$ (**$+35.83\%$ reduction**, $p < 10^{-300}$). |
| 11 | **Acquisition budget is lower than baseline** | Delay summation | **VERIFIED** | $\sum \text{PLD} = 6.50\text{ s}$ vs. Baseline $9.10\text{ s}$ (**$28.6\%$ faster**). |
| 12 | **Training curves are healthy and stable** | Actual history inspection | **VERIFIED** | Monotonic loss convergence; best checkpoints match early stopping records. |

---

## 10. Explicit Answers to Prompt Questions

1. **Are the reported 16–17% CBF RMSE and 40% ATT RMSE improvements reproducible from saved predictions?**  
   **YES.** Independent recalculation yields exactly $+16.76\%$ for the optimized DNN and $+17.23\%$ for the optimized Feature ELM on CBF RMSE ($3.7266 \to 3.0846$), and $+40.47\%$ for the optimized DNN and $+40.43\%$ for the optimized Feature ELM on ATT RMSE ($0.30580\text{ s} \to 0.18206\text{ s}$). Numerical discrepancies vs. the reported CSVs are under $10^{-15}$.

2. **Are the much larger MAE improvements also reproducible?**  
   **YES.** Global CBF MAE improved by $+35.83\%$ ($2.5323 \to 1.6249$), and global ATT MAE improved by $+63.33\%$ ($0.19352\text{ s} \to 0.07097\text{ s}$). In the early-arrival interval $[0.5, 1.0)\text{ s}$, ATT MAE improved by $+81.68\%$ ($0.3691\text{ s} \to 0.0676\text{ s}$).

3. **Are the optimized training curves healthy and consistent with the saved checkpoints?**  
   **YES.** The training curves exhibit smooth, stable decay. ATTnet achieved its minimum validation loss at Epoch 20 ($0.060993$) and early-stopped at Epoch 30; CBFnet reached its minimum validation loss at Epoch 10 ($0.011518$) and early-stopped at Epoch 20. Both saved `.pth` checkpoints match the recorded best validation weights.

4. **Is the early-arrival ATT improvement genuine?**  
   **YES.** The baseline DNN completely failed for $\text{ATT} < 1.0\text{ s}$, predicting an invariant mean value near $\sim 1.1\text{ s}$ and never outputting any value below $0.9314\text{ s}$. The optimized acquisition enables genuine physical transit tracking down to $0.5000\text{ s}$, eliminating $86.6\%$ of the bias and increasing CCC from $0.0047$ to $0.4569$.

5. **Is the acquisition-design explanation supported by the actual forward model?**  
   **YES.** When all PLDs are greater than ATT, all measurements fall into the post-bolus clearance regime, where the signal factorizes into a fixed 1D vector multiplied by a scalar, producing identical gradients ($\cos\theta \equiv 1.000000, \kappa \approx 4 \times 10^{16}$). Introducing $\text{PLD}_1 = 0.40\text{ s}$ captures the bolus during arrival, breaking collinearity ($\cos\theta \le 0.70$) and restoring rank-2 identifiability ($\kappa \le 85, \sigma_2 \approx 15$).

6. **Is the result ready to be presented as a research finding, and what limitations must be disclosed?**  
   **YES, the finding is scientifically rigorous and ready for publication.**  
   *Mandatory Disclosures/Limitations:*
   - The findings are based on synthetic signals generated from the Buxton General Kinetic Model with dual-channel Rician noise.
   - Real in vivo ASL data contains additional confounding factors: patient motion, partial volume averaging, $B_1^+$ transmit inhomogeneity, and macrovascular transit artifacts.
   - Clinical sequences must ensure background suppression inversion pulses are compatible with the short initial delay ($\text{PLD}_1 = 0.40\text{ s}$).
   - The performance gain reflects the combined effect of an optimized acquisition protocol and a matched neural estimator, rather than a standalone algorithmic innovation on fixed data.
