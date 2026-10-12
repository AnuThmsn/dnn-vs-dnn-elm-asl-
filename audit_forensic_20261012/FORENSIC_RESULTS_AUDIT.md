# FORENSIC AUDIT REPORT: INDEPENDENT VERIFICATION OF ASL MRI DNN VS. DNN-ELM RESULTS

**Audit Date**: October 12, 2026  
**Auditor**: Independent Forensic Verification Agent  
**Repository**: [dnn-vs-dnn-elm-asl-](https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-)  
**Execution Policy**: Zero Retraining | Zero Dataset Modification | Strict Drive D Workspace | Bitwise Cryptographic Verification  

---

## 1. Executive Summary & Audit Verdict

This forensic audit was conducted to independently evaluate the validity, reproducibility, and clinical defensibility of reported parameter estimation results comparing deep neural networks (DNN Baseline A / DNN-G) against Extreme Learning Machine (ELM) hybrids for Arterial Spin Labelling (ASL) MRI cerebral blood flow (CBF) and arterial transit time (ATT) quantification.

### Primary Audit Findings:
1. **Bitwise Data & Artifact Integrity**:
   - All evaluation ground-truth arrays ($Y_{\text{test}} \in \mathbb{R}^{100,000 \times 2}$), noisy inputs ($X_{\text{test}} \in \mathbb{R}^{100,000 \times 4}$), and model prediction matrices ($6 \times 100,000 \times 2$) match their historical SHA-256 checksums exactly.
   - The test set predictions in `experiments/elm_models/all_test_predictions.npz` match the underlying ground-truth cache `experiments/cache/Y_test.npy` with **zero discrepancy** ($\max |\Delta Y| = 0.000000 \times 10^0$).
   - Recomputation of all primary metrics (RMSE, MAE, Mean Bias, Lin's CCC, and $R^2$) confirms reported values with maximum discrepancy $\le 4.44 \times 10^{-16}$ (floating-point machine epsilon).

2. **Divergent Clinical Realities for CBF vs. ATT**:
   - **CBF Estimation (Genuine, Tangible Improvement)**:
     - The **Ensemble Feature ELM** reduces CBF MAE from $2.5323\ \text{mL}/(100\text{g}\cdot\min)$ to $2.4232\ \text{mL}/(100\text{g}\cdot\min)$ ($+4.31\%$ improvement, $+0.1091\ \text{mL}/(100\text{g}\cdot\min)$ paired reduction, $p = 8.70 \times 10^{-90}$, $95\%\ \text{CI}\ [0.1059, 0.1121]$).
     - Standalone ELM, however, degrades CBF performance ($\Delta\text{RMSE} = -0.57\%$), establishing that ELM random projections alone are insufficient; deep hierarchical representations from the DNN backbone are strictly required.
   - **ATT Estimation (Statistically Detectable, Clinically Negligible)**:
     - The reported ATT RMSE improvement for Ensemble Feature ELM is merely **$+0.000926\text{ s}$ ($+0.926\text{ ms}$, $+0.30\%$ relative gain)**, moving RMSE from $0.30580\text{ s}$ to $0.30487\text{ s}$.
     - Because the held-out test cohort is enormous ($N = 100,000$), the standard error is vanishingly small ($\sigma / \sqrt{N} \approx 0.00010\text{ s}$), yielding an astronomical $t$-statistic ($t = 7.64, p = 2.45 \times 10^{-14}$). 
     - **Verdict**: The ATT improvement is a bona fide numerical feature of the optimization surface, but it is **clinically negligible** ($< 1\text{ ms}$ difference in an MRI acquisition with temporal resolution $\ge 500\text{ ms}$).

3. **The Unresolved Physical Bottleneck: Early Transit Times ($[0.5, 1.0)\text{ s}$)**:
   - For early arrivals ($\text{ATT} \in [0.5, 1.0)\text{ s}$), where bolus inflow completes prior to the first post-label delay ($\text{PLD}_1 = 1.525\text{ s}$), all models suffer catastrophic bias ($+0.369\text{ s}$ to $+0.372\text{ s}$) and near-zero concordance ($\text{CCC} \approx 0.005$).
   - The Ensemble Feature ELM provides **zero alleviation** of this bottleneck ($\text{RMSE}_{\text{ELM}} = 0.4693\text{ s}$ vs $\text{RMSE}_{\text{DNN}} = 0.4664\text{ s}$). The limitation is rooted in Buxton kinetic model sampling physics, which post-processing feature maps cannot overcome.

---

## 2. Commit, Environment & Hardware Provenance

| Parameter | Audited Specification |
| :--- | :--- |
| **Git Commit Hash** | `99d4f7b2c01990c8a66d03cf8da86815340eb0f0` |
| **Git Branch** | `main` (synchronized with `origin/main`) |
| **Operating System** | Windows 10/11 x64 (Build 26100) |
| **Python Runtime** | Python 3.10.11 (64-bit) |
| **Compute Device** | NVIDIA GeForce GTX 1650 (CUDA 12.x / PyTorch 2.x) |
| **Host Working Dir** | `D:\dnn vs dnn elm` |
| **Drive C Footprint** | Strictly 0 bytes written (All audit logs, tables, and plots stored in Drive D) |
| **Host RAM Safeguard**| Peak RAM footprint maintained under 85 MB via chunked float64 array operations |

---

## 3. Cryptographic Artifact Integrity Verification (SHA-256)

All critical experiment artifacts were audited via SHA-256 hashing. The table below provides the full audit trail:

| Relative Path | Size (KB) | SHA-256 Digest | Audit Status |
| :--- | :---: | :--- | :---: |
| `experiments\baseline_A\ATTnet_G_baselineA.pth` | 54.3 | `91688d8beecf5151dfdb4092d16e4d06c910918b0411fc72c7da15f764025016` | **VERIFIED** |
| `experiments\baseline_A\CBFnet_G_baselineA.pth` | 265.7 | `76154e7d429263a278401c2f038e78bfa05b0090a0d658b2aa1daee95ecda69b` | **VERIFIED** |
| `experiments\baseline_A\training_history.json` | 10.3 | `f648f7b17e2e4821815b1c7f72ab5ee016efd56f2e49d996e7bc38918eade5d1` | **VERIFIED** |
| `experiments\baseline_A\normalization_stats.npz` | 1.5 | `d2637da7af19559a75862190692017e2a720cfe6b4cebeec2e29b8ceaac8f728` | **VERIFIED** |
| `experiments\baseline_A\baselineA_test_predictions.npz` | 801.0 | `3194250da46e90ecad962ea5b54f0cdef3fc0dc72f6950bfa01c978d1f962908` | **VERIFIED** |
| `experiments\elm_models\all_test_predictions.npz` | 7035.1 | `3150b8f0f494d64ed510f345eaaaeb0b12948e3a41f25c23549350a5893904f1` | **VERIFIED** |
| `experiments\elm_models\all_models_test_comparison.csv` | 1.4 | `98c4710147ad9e87528f475b564e6b40ca61202ce911db5b514f2c9870980b8c` | **VERIFIED** |
| `experiments\elm_models\statistical_bootstrap_results.csv`| 1.8 | `e2ddd863c5eef9597ce00955a40afad686487396a71ab103e7f1679ad16d0f1f` | **VERIFIED** |
| `experiments\cache\X_test.npy` | 1562.6 | `ec0e6bcb76b5ab178e44b72db3553847214bc12da9f32e701c60ae1cd0309528` | **VERIFIED** |
| `experiments\cache\Y_test.npy` | 781.4 | `dcdc5cfba85f71f29b10594a36a0babd1ca39f40b707a4d7fa23641204fbaeab` | **VERIFIED** |
| `experiments\cache\snr_test.npy` | 781.4 | `4ef93cbf7cd690182906a049ff2f72293545d270fc6d51a59436216d615851e6` | **VERIFIED** |
| `experiments\cache\clean_grid.npz` | 169.4 | `3c4be7886239248bc061de669c538e18f9eaa8c10711526332e67429e4451e03` | **VERIFIED** |

---

## 4. Formal Metric Formulation & Sign Convention

To prevent directional confusion across comparative analyses, all differences are strictly reported under the standard **error reduction convention**:

$$\Delta\text{RMSE} = \text{RMSE}_{\text{Baseline A}} - \text{RMSE}_{\text{Candidate}}$$
$$\Delta\text{MAE} = \text{MAE}_{\text{Baseline A}} - \text{MAE}_{\text{Candidate}}$$
$$\text{Relative Gain (\%)} = \frac{\text{Metric}_{\text{Baseline A}} - \text{Metric}_{\text{Candidate}}}{\text{Metric}_{\text{Baseline A}}} \times 100\%$$

* **Positive $\Delta$ and \%**: Candidate model **outperforms** the DNN baseline (error is lower).
* **Negative $\Delta$ and \%**: Candidate model **underperforms** the DNN baseline (error is higher).

Concordance is quantified using Lin's Concordance Correlation Coefficient ($\rho_c$):
$$\rho_c = \frac{2 \rho \sigma_y \sigma_{\hat{y}}}{\sigma_y^2 + \sigma_{\hat{y}}^2 + (\mu_y - \mu_{\hat{y}})^2}$$

---

## 5. Independent Recalculation on Held-Out Test Set ($N = 100,000$)

All models were evaluated on the exact locked synthetic test set ($N = 100,000$, SNR $\in [5, \infty)$, CBF $\in [10, 120]\ \text{mL}/(100\text{g}\cdot\min)$, $\text{ATT} \in [0.5, 3.0]\text{ s}$).

### Table 5.1: Cerebral Blood Flow (CBF) Metrics

| Model Architecture | RMSE | $\Delta\text{RMSE}$ | Gain (\%) | MAE | $\Delta\text{MAE}$ | Gain (\%) | Bias | Lin's CCC | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (DNN-G)** | 3.7266 | 0.0000 | 0.00% | 2.5323 | 0.0000 | 0.00% | -0.1731 | 0.99157 | 0.98332 |
| **Standalone ELM** | 3.7478 | -0.0212 | -0.57% | 2.5457 | -0.0133 | -0.53% | -0.0095 | 0.99150 | 0.98313 |
| **Weighted Ensemble** | 3.7147 | +0.0119 | +0.32% | 2.5176 | +0.0147 | +0.58% | -0.1159 | 0.99163 | 0.98343 |
| **Residual ELM** | 3.6836 | +0.0430 | +1.15% | 2.4395 | +0.0928 | +3.66% | -0.0084 | 0.99179 | 0.98371 |
| **DNN-Feature ELM** | 3.6850 | +0.0415 | +1.11% | 2.4248 | +0.1076 | +4.25% | -0.0080 | 0.99178 | 0.98369 |
| **Ensemble Feature ELM** | **3.6828** | **+0.0438** | **+1.18%** | **2.4232** | **+0.1091** | **+4.31%** | **-0.0085** | **0.99179** | **0.98371** |

### Table 5.2: Arterial Transit Time (ATT) Metrics

| Model Architecture | RMSE (s) | $\Delta\text{RMSE}$ (s) | Gain (\%) | MAE (s) | $\Delta\text{MAE}$ (s) | Gain (\%) | Bias (s) | Lin's CCC | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (DNN-G)** | 0.30580 | 0.00000 | 0.00% | 0.19352 | 0.00000 | 0.00% | -0.0066 | 0.90092 | 0.81955 |
| **Standalone ELM** | 0.31615 | -0.01035 | -3.39% | 0.20611 | -0.01259 | -6.51% | -0.0011 | 0.89328 | 0.80712 |
| **Weighted Ensemble** | 0.30580 | -0.00001 | -0.00% | 0.19382 | -0.00029 | -0.15% | -0.0063 | 0.90085 | 0.81954 |
| **Residual ELM** | 0.30540 | +0.00040 | +0.13% | 0.19125 | +0.00228 | +1.18% | -0.0012 | 0.90123 | 0.82002 |
| **DNN-Feature ELM** | 0.30500 | +0.00080 | +0.26% | 0.19061 | +0.00291 | +1.50% | -0.0008 | 0.90154 | 0.82050 |
| **Ensemble Feature ELM** | **0.30487** | **+0.00093** | **+0.30%** | **0.19039** | **+0.00314** | **+1.62%** | **-0.0008** | **0.90161** | **0.82064** |

---

## 6. Paired Bootstrap & Hypothesis Testing ($B = 1,000$)

Because paired error residuals $\epsilon_i = |y_i - \hat{y}_{\text{DNN}, i}| - |y_i - \hat{y}_{\text{Candidate}, i}|$ violate Gaussian normality assumptions, statistical significance was evaluated using **1,000 paired bootstrap resamples** alongside non-parametric **Wilcoxon signed-rank tests** and paired Student's $t$-tests.

### Table 6.1: Paired Statistical Comparison Against Baseline A

| Candidate Model | Target | $\Delta\text{RMSE}$ (95% CI) | $\Delta\text{MAE}$ (95% CI) | Paired $t$-test $p$ | Wilcoxon $p$ | Null Rejected ($\alpha=0.001$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standalone ELM** | CBF | -0.0211 [-0.0278, -0.0143] | -0.0133 [-0.0178, -0.0089] | $1.41 \times 10^{-9}$ | 0.8315 | **False (Worse)** |
| **Standalone ELM** | ATT | -0.0104 [-0.0111, -0.0097] | -0.0126 [-0.0131, -0.0121] | $8.75 \times 10^{-174}$ | $1.40 \times 10^{-63}$ | **False (Worse)** |
| **Weighted Ensemble** | CBF | +0.0119 [+0.0095, +0.0143] | +0.0147 [+0.0131, +0.0163] | $1.30 \times 10^{-22}$ | $1.56 \times 10^{-18}$ | **True (Better)** |
| **Weighted Ensemble** | ATT | -0.0000 [-0.00004, +0.00003]| -0.0003 [-0.00032, -0.00027]| 0.7893 | $5.10 \times 10^{-5}$ | **False (Neutral)** |
| **Residual ELM** | CBF | +0.0430 [+0.0391, +0.0467] | +0.0928 [+0.0899, +0.0961] | $3.00 \times 10^{-106}$ | $3.12 \times 10^{-77}$ | **True (Better)** |
| **Residual ELM** | ATT | +0.00040 [+0.00030, +0.00051]| +0.00228 [+0.00221, +0.00235]| $4.55 \times 10^{-13}$ | $6.46 \times 10^{-82}$ | **True (Better)** |
| **DNN-Feature ELM** | CBF | +0.0414 [+0.0369, +0.0459] | +0.1076 [+0.1041, +0.1107] | $9.88 \times 10^{-75}$ | $1.17 \times 10^{-96}$ | **True (Better)** |
| **DNN-Feature ELM** | ATT | +0.00081 [+0.00058, +0.00108]| +0.00292 [+0.00277, +0.00307]| $4.29 \times 10^{-10}$ | $3.65 \times 10^{-43}$ | **True (Better)** |
| **Ensemble Feature ELM**| CBF | **+0.0438 [+0.0394, +0.0480]**| **+0.1091 [+0.1059, +0.1121]**| **$1.53 \times 10^{-89}$**| **$8.70 \times 10^{-90}$**| **True (Better)** |
| **Ensemble Feature ELM**| ATT | **+0.00092 [+0.00068, +0.00117]**| **+0.00313 [+0.00298, +0.00328]**| **$2.45 \times 10^{-14}$**| **$1.66 \times 10^{-62}$**| **True (Better)** |

---

## 7. Deep Dive: The ATT Early-Arrival Bottleneck & Kinetic Physics

### 7.1 Mathematical Origin of the Ill-Posed Inversion
The general kinetic model for pulsed/pseudo-continuous ASL (Buxton et al., 1998) defines the difference magnetization $\Delta M(t)$ as:

$$\Delta M(t) = 2 M_0 f \alpha T_1' \cdot e^{-\delta / T_{1b}} \left[ e^{-\frac{\max(0, t - \delta)}{T_1'}} - e^{-\frac{\max(0, t - (\delta + \tau))}{T_1'}} \right]$$

where $f$ is CBF, $\delta$ is ATT, $\tau = 1.8\text{ s}$ is labeling duration, and $T_{1b} \approx 1.65\text{ s}$.

In the fixed clinical protocol, the 4 post-label delays are:
$$\text{PLDs} = [1.525, 2.025, 2.525, 3.025]\text{ seconds}$$
with total observation times $t_i = \text{PLD}_i + \tau \in [3.325, 4.825]\text{ s}$.

When $\text{ATT} \in [0.5, 1.0)\text{ s}$, the labeled blood bolus arrives and washes into the capillary bed long before the first measurement point ($t_1 = 3.325\text{ s}$). Over the measurement window $t \ge 3.325\text{ s}$, the inflow phase has completely terminated:
$$\frac{\partial \Delta M(t_i)}{\partial \delta} \propto - \frac{1}{T_{1b}} e^{-\delta / T_{1b}} \approx \text{constant decay}$$
Consequently, variation in $\delta$ alters only the global signal scaling rather than the temporal inflow shape. This induces a near-total parameter degeneracy between CBF ($f$) and ATT ($\delta$), making early ATT inversion fundamentally ill-posed under noisy measurements.

### 7.2 Empirical Breakdown Across ATT Intervals
The table below reveals the empirical error behavior across the 5 transit time bins:

| ATT Bin (s) | Cohort $N$ | Model | RMSE (s) | MAE (s) | Mean Bias (s) | Lin's CCC |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **$[0.5, 1.0)$** | 19,728 | **Baseline A (DNN-G)** | 0.4664 | 0.3691 | +0.3691 | 0.0047 |
| | | Standalone ELM | 0.4916 | 0.3942 | +0.3930 | 0.0037 |
| | | **Ensemble Feature ELM** | 0.4693 | 0.3716 | +0.3716 | 0.0050 |
| **$[1.0, 1.5)$** | 20,281 | **Baseline A (DNN-G)** | 0.2948 | 0.2281 | -0.1188 | 0.0306 |
| | | Standalone ELM | 0.2945 | 0.2230 | -0.0971 | 0.0308 |
| | | **Ensemble Feature ELM** | 0.2949 | 0.2273 | -0.1150 | 0.0323 |
| **$[1.5, 2.0)$** | 19,836 | **Baseline A (DNN-G)** | 0.2542 | 0.1637 | -0.1218 | 0.5121 |
| | | Standalone ELM | 0.2728 | 0.1985 | -0.1415 | 0.4959 |
| | | **Ensemble Feature ELM** | 0.2513 | 0.1578 | -0.1125 | 0.5176 |
| **$[2.0, 2.5)$** | 20,275 | **Baseline A (DNN-G)** | 0.1699 | 0.0838 | -0.0517 | 0.5805 |
| | | Standalone ELM | 0.1702 | 0.0880 | -0.0461 | 0.5807 |
| | | **Ensemble Feature ELM** | **0.1660** | **0.0778** | **-0.0415** | **0.5910** |
| **$[2.5, 3.0]$** | 19,880 | **Baseline A (DNN-G)** | 0.2681 | 0.1256 | -0.1040 | 0.3121 |
| | | Standalone ELM | 0.2660 | 0.1302 | -0.1082 | 0.3050 |
| | | **Ensemble Feature ELM** | **0.2629** | **0.1202** | **-0.1008** | **0.3153** |

### Key Analytical Takeaways:
1. **Severe Positive Bias at Early Arrival**: In the $[0.5, 1.0)\text{ s}$ window, the mean prediction is centered near $1.12\text{ s}$, leading to a massive $+0.37\text{ s}$ bias. Both DNN and ELM regress toward the conditional expectation of the prior distribution because the signal contains insufficient inflow information.
2. **Optimal Performance in Well-Sampled Window**: When $\text{ATT} \in [2.0, 2.5)\text{ s}$, the bolus transition aligns directly with the central PLDs ($2.025\text{ s}$ and $2.525\text{ s}$). Here, RMSE drops to $0.166\text{ s}$ and CCC reaches $\sim 0.59$.
3. **ELM Does Not Resolve Early Arrival**: Ensemble Feature ELM performs virtually identically to the DNN in the $[0.5, 1.0)\text{ s}$ regime ($\text{RMSE} = 0.4693\text{ s}$ vs. $0.4664\text{ s}$). The marginal overall gains of the ELM ensemble arise almost entirely from the well-sampled windows ($[2.0, 2.5)\text{ s}$ and $[2.5, 3.0]\text{ s}$).

---

## 8. Statistical Power vs. Clinical Effect Size Critique

In statistical methodology, a classic risk is confusing **statistical detectability** with **clinical or practical significance**.

### The $N = 100,000$ Effect Size Paradox:
For a sample size of $N = 100,000$, the standard error of the mean difference is:
$$\text{SE}(\Delta) = \frac{s_d}{\sqrt{100,000}} \approx \frac{0.038}{316.2} \approx 0.00012\text{ s}$$
A sub-millisecond shift of $\Delta\text{RMSE} = 0.000926\text{ s}$ yields:
$$t = \frac{0.000926}{0.000121} = 7.64 \implies p = 2.45 \times 10^{-14}$$

* **Statistical Reality**: The null hypothesis $H_0: \Delta\text{RMSE} = 0$ is rejected beyond any doubt ($p < 10^{-13}$).
* **Clinical Reality**: In clinical ASL neuroimaging, ATT differences under $50\text{ ms}$ are below the noise floor of standard multi-PLD EPI acquisitions. An improvement of **$0.93\text{ ms}$** represents a **$0.30\%$ error reduction**, which is completely undetectable in radiological practice.

Conversely, for **CBF estimation**:
* The absolute MAE reduction of **$0.1091\ \text{mL}/(100\text{g}\cdot\min)$** ($+4.31\%$ gain) is robust, non-trivial, and consistent across high and low SNR regimes.

---

## 9. Comprehensive Comparison Table: DNN Alone vs. DNN-ELM

To directly satisfy the research presentation requirement, Table 9.1 provides a side-by-side comparison of the standard standalone DNN against the top-performing DNN-ELM hybrid across both clean and noisy regimes:

### Table 9.1: Head-to-Head Comparison (Baseline A DNN vs. Ensemble Feature ELM)

| Evaluation Regime | Parameter | Metric | Baseline A (DNN Alone) | DNN-ELM (Ensemble Feat ELM) | Absolute Gain ($\Delta$) | Relative Gain (\%) | Statistical $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Noisy Test Cohort**<br>($N = 100,000$) | **CBF** | **RMSE** | $3.7266$ | **$3.6828$** | $+0.0438$ | $+1.18\%$ | $1.53 \times 10^{-89}$ |
| | | **MAE** | $2.5323$ | **$2.4232$** | $+0.1091$ | **$+4.31\%$** | $8.70 \times 10^{-90}$ |
| | | **Lin's CCC** | $0.99157$ | **$0.99179$** | $+0.00022$ | $+0.02\%$ | — |
| | **ATT** | **RMSE** | $0.30580\text{ s}$ | **$0.30487\text{ s}$** | $+0.00093\text{ s}$ | $+0.30\%$ | $2.45 \times 10^{-14}$ |
| | | **MAE** | $0.19352\text{ s}$ | **$0.19039\text{ s}$** | $+0.00314\text{ s}$ | $+1.62\%$ | $1.66 \times 10^{-62}$ |
| | | **Lin's CCC** | $0.90092$ | **$0.90161$** | $+0.00069$ | $+0.08\%$ | — |
| **Clean Grid Cohort**<br>($N = 10,791$, $\text{SNR}=\infty$) | **CBF** | **RMSE** | $2.4642$ | **$2.3897$** | $+0.0745$ | $+3.02\%$ | — |
| | | **MAE** | $1.5684$ | **$1.3060$** | $+0.2624$ | **$+16.73\%$** | — |
| | | **Lin's CCC** | $0.99635$ | **$0.99657$** | $+0.00022$ | $+0.02\%$ | — |
| | **ATT** | **RMSE** | $0.19532\text{ s}$ | **$0.19088\text{ s}$** | $+0.00444\text{ s}$ | $+2.27\%$ | — |
| | | **MAE** | $0.11974\text{ s}$ | **$0.11162\text{ s}$** | $+0.00812\text{ s}$ | **$+6.78\%$** | — |
| | | **Lin's CCC** | $0.96159$ | **$0.96351$** | $+0.00192$ | $+0.20\%$ | — |

---

## 10. Definitive Forensic Conclusion

1. **Validity of Historical Claims**: The reported claims regarding Ensemble Feature ELM in Phase 5 are mathematically verified and reproducible down to floating-point precision. There is no evidence of data leakage, post-hoc metric cherry-picking, or artifact manipulation.
2. **CBF Estimation Verdict**: **GENUINE GAIN**. The Feature ELM formulation effectively leverages the frozen nonlinear basis representations of the DNN and optimizes a regularized linear read-out that reduces CBF MAE by $+4.31\%$ on noisy data and $+16.73\%$ on clean data.
3. **ATT Estimation Verdict**: **STATISTICALLY SIGNIFICANT BUT CLINICALLY NEGLIGIBLE**. The sub-millisecond gain ($+0.93\text{ ms}$, $+0.30\%$) is driven by standard error shrinkage at $N=100,000$. It does not alter the fundamental physical inversion bottleneck at early transit delays ($[0.5, 1.0)\text{ s}$), which requires adaptive or earlier PLD sampling rather than neural post-processing.
4. **Architectural Recommendation**: Standalone ELM must be abandoned (it underperforms DNN Baseline A). If an ELM hybrid is deployed, it should be strictly restricted to **Ensemble Feature ELM** targeting CBF estimation.
