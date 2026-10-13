# Final Scientific Presentation Report: Comparative ASL-MRI Parameter Estimation (DNN vs. DNN-Feature ELM)

**Project:** Comparative Deep Neural Network vs. Extreme Learning Machine Parameter Estimation for Arterial Spin Labeling MRI  
**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Presentation Directory:** [`results/BEST_RESULTS_PRESENTATION/`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/)  
**Execution Date:** 2026-10-13  
**Status:** Presentation-Ready & Fully Reproducible  

---

## 1. Project Objective & Fixed Acquisition Protocol

The primary objective of this project is to perform a scientifically rigorous, transparent, and reproducible comparison between a standard Deep Neural Network (DNN-G) and an Extreme Learning Machine hybrid (DNN-Feature ELM) for simultaneous cerebral blood flow (CBF) and arterial transit time (ATT) quantification in pseudo-continuous Arterial Spin Labeling (pCASL) MRI.

To ensure absolute benchmark integrity and avoid artificial metric manipulation, the acquisition protocol and simulation parameters are strictly frozen across all evaluations:
- **Post-Labeling Delays (PLDs):** $[1.525, 2.025, 2.525, 3.025]$ seconds
- **Label Duration ($\tau$):** $1.80$ seconds
- **Physical Reference Constants:**
  - Blood Longitudinal Relaxation ($T_{1,\text{blood}}$): $1.66\text{ s}$
  - Tissue Longitudinal Relaxation ($T_{1,\text{tissue}}$): $1.20\text{ s}$
  - Inversion Efficiency ($\alpha$): $0.85$
  - Transit Attenuation ($\beta$): $0.75$
  - Blood-Brain Partition Coefficient ($\lambda$): $0.90\text{ mL/g}$
- **Noise Model:** Dual-channel Rician noise across 12 standard SNR levels ($5.0\text{ dB}$ to $\infty\text{ dB}$)
- **Target Physiological Ranges:**
  - $\text{CBF} \in [0.0, 100.0]\text{ mL}/(100\text{g}\cdot\text{min})$
  - $\text{ATT} \in [0.5, 3.0]\text{ seconds}$

---

## 2. Model Architectures & Estimation Principles

### 2.1 Baseline A: Standardized Deep Neural Network (DNN-G)
- **CBF Network:** 5-layer fully connected architecture with 128 units per hidden layer, Exponential Linear Unit (ELU) activations, and a single linear output layer.
- **ATT Network:** 4-layer fully connected architecture with 64 units per hidden layer, ELU activations, and a single linear output layer.
- **Training Principle:** Standard mini-batch Adam optimization trained on $2,000,000$ synthetic observations to minimize MSE. Checkpoints are strictly frozen at [`experiments/baseline_A/CBFnet_G_baselineA.pth`](file:///d:/dnn%20vs%20dnn%20elm/experiments/baseline_A/CBFnet_G_baselineA.pth) and [`experiments/baseline_A/ATTnet_G_baselineA.pth`](file:///d:/dnn%20vs%20dnn%20elm/experiments/baseline_A/ATTnet_G_baselineA.pth).

### 2.2 Primary Candidate: 5-Seed Ensemble Feature ELM
- **Principle:** Instead of discarding the frozen DNN, the Feature ELM feeds normalized 4-PLD signals into the frozen DNN backbone and taps the penultimate hidden representation ($h_5 \in \mathbb{R}^{128}$ for CBF; $h_4 \in \mathbb{R}^{64}$ for ATT).
- **Readout Architecture:** The penultimate vector is mapped through a random projection layer ($512$ hidden units with $\tanh$ activation) and solved via closed-form regularized least squares:
  $$\beta = (H^T H + \alpha I)^{-1} H^T Y$$
  with ridge parameter $\alpha = 10^{-2}$.
- **Ensemble Averaging:** Predictions are averaged across 5 independent random initializations (seeds $42, 43, 44, 45, 46$) to cancel random weight fluctuations.
- **Compute Advantage:** Closed-form solution executes in $\approx 15\text{ seconds}$ on $1,000,000$ samples using float64 GPU matrix accumulation, requiring $< 80\text{ MB}$ memory and zero iterative backpropagation.

---

## 3. Experimental Fairness & Evaluation Protocol

1. **Zero Data Leakage:** Preprocessing statistics, feature scalers, and hyperparameter tuning were conducted strictly on the training set ($N=1,000,000$) and validation set ($N=100,000$). The locked test cohort was accessed exclusively for final evaluation.
2. **Identical Test Cohort:** Both models were evaluated on the exact same $100,000$ locked test observations, identical synthetic ground truth, and identical Rician noise realizations.
3. **Paired Statistical Resampling:** Confidence intervals were computed via $B=2,000$ bootstrap resamples paired by individual test case index.
4. **Honest Sign Convention:** Percentage improvement is defined such that positive values indicate candidate superiority:
   $$\text{Gain}_{\%} = 100 \times \frac{\text{RMSE}_{\text{DNN}} - \text{RMSE}_{\text{ELM}}}{\text{RMSE}_{\text{DNN}}}$$

---

## 4. Authoritative Performance Comparison Table

Evaluated on the locked test cohort ($N=100,000$):

| Parameter | Metric | Baseline A (DNN-G) | Ensemble Feature ELM | Absolute Difference ($\Delta$) | Percentage Improvement (%) | Paired 95% Bootstrap CI | Paired $t$-test $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CBF** | **RMSE** ($\text{mL}/100\text{g}/\text{min}$) | 3.7266 | **3.6849** | $+0.0417$ | **+1.12%** | $[+0.0378, +0.0457]$ | $4.28 \times 10^{-97}$ |
| **CBF** | **MAE** ($\text{mL}/100\text{g}/\text{min}$) | 2.5323 | **2.4296** | $+0.1027$ | **+4.06%** | $[+0.0965, +0.1089]$ | $1.15 \times 10^{-112}$ |
| **CBF** | **Bias** ($\text{mL}/100\text{g}/\text{min}$) | -0.1731 | **-0.0080** | $+0.1651$ | **+95.38%** | — | — |
| **CBF** | **CCC** | 0.9916 | **0.9918** | $+0.0002$ | +0.02% | — | — |
| **CBF** | **$R^2$** | 0.9833 | **0.9837** | $+0.0004$ | +0.04% | — | — |
| **ATT** | **RMSE** ($\text{seconds}$) | 0.3058 | **0.3048** | $+0.0010$ | **+0.31%** | $[+0.00075, +0.00115]$ | $5.56 \times 10^{-20}$ |
| **ATT** | **MAE** ($\text{seconds}$) | 0.1935 | **0.1905** | $+0.0030$ | **+1.55%** | $[+0.00262, +0.00338]$ | $2.14 \times 10^{-42}$ |
| **ATT** | **Bias** ($\text{seconds}$) | -0.0066 | **-0.0008** | $+0.0058$ | **+87.88%** | — | — |
| **ATT** | **CCC** | 0.9009 | **0.9016** | $+0.0007$ | +0.08% | — | — |
| **ATT** | **$R^2$** | 0.8195 | **0.8207** | $+0.0012$ | +0.15% | — | — |
| **Early ATT** | **RMSE** $[0.5, 1.0)\text{s}$ | **0.4664** | 0.4694 | $-0.0030$ | **-0.64%** (Degraded) | $[-0.00412, -0.00188]$ | $8.45 \times 10^{-08}$ |
| **Early ATT** | **Bias** $[0.5, 1.0)\text{s}$ | **+0.3691** | +0.3717 | $-0.0026$ | **-0.70%** (Degraded) | — | — |

---

## 5. Main Graphs & Detailed Interpretation

All figures are located in [`results/BEST_RESULTS_PRESENTATION/graphs/`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/):

### Figure 1: CBF & ATT RMSE Comparison ([`01_cbf_rmse_comparison.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/01_cbf_rmse_comparison.png) & [`02_att_rmse_comparison.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/02_att_rmse_comparison.png))
- **Interpretation:** The Feature ELM achieves a modest CBF RMSE reduction of $0.0417\text{ mL}/(100\text{g}\cdot\text{min})$ ($+1.12\%$). For ATT, the absolute reduction is exactly $0.0010\text{ seconds}$ ($1\text{ millisecond}$, or $+0.31\%$). The axes are deliberately drawn starting from zero to honestly reflect scale and prevent visual exaggeration.

### Figure 2: Percentage Improvement Summary ([`05_percentage_improvement.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/05_percentage_improvement.png))
- **Interpretation:** This plot balances the positive gains against the regression:
  - CBF RMSE: $+1.12\%$
  - CBF MAE: $+4.06\%$
  - ATT RMSE: $+0.31\%$
  - ATT MAE: $+1.55\%$
  - Early-ATT RMSE ($[0.5, 1.0)\text{ s}$): **$-0.64\%$** (negative bar below zero baseline).

### Figure 3: Scatter & Calibration Fidelity ([`06_cbf_prediction_scatter.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/06_cbf_prediction_scatter.png) & [`07_att_prediction_scatter.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/07_att_prediction_scatter.png))
- **Interpretation:** The 5,000-point sampled scatter shows that the Feature ELM aligns closely with the identity line ($y = x$). For CBF, the baseline DNN's slight underestimation at high perfusion values is resolved, shifting mean bias from $-0.173$ to $-0.008$. For ATT, both models display identical dispersion patterns.

### Figure 4: Error Residual Distributions ([`08_error_distribution.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/08_error_distribution.png))
- **Interpretation:** Error histograms reveal that the Feature ELM distribution is almost perfectly centered around zero, whereas the DNN-G residual distribution exhibits a small negative skew.

### Figure 5: SNR Robustness ([`09_snr_robustness.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/09_snr_robustness.png))
- **Interpretation:** Across all 12 SNR levels, the Feature ELM and DNN curves track each other tightly. The ELM's relative gain is largest at high SNR ($> 30\text{ dB}$, where noise is minimal) and shrinks to negligible levels ($< 0.2\%$) at low SNR ($5\text{--}10\text{ dB}$).

### Figure 6: ATT Physiological Interval Breakdown ([`10_att_interval_performance.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/10_att_interval_performance.png))
- **Interpretation:** For normal cortical and delayed transit ($\text{ATT} \ge 1.5\text{ s}$), both models achieve low RMSE ($0.16\text{--}0.26\text{ s}$). In the early window ($[0.5, 1.0)\text{ s}$), errors rise to $0.47\text{ s}$ for both models. The ELM performs slightly worse than DNN here ($0.4694\text{ s}$ vs $0.4664\text{ s}$).

### Figure 7: Paired Error Difference Distribution ([`11_paired_error_difference.png`](file:///d:/dnn%20vs%20dnn%20elm/results/BEST_RESULTS_PRESENTATION/graphs/11_paired_error_difference.png))
- **Interpretation:** Plotting the sample-wise absolute error reduction ($|e_{\text{DNN}}| - |e_{\text{ELM}}|$) confirms a positive mean shift of $+0.1027\text{ mL}/(100\text{g}\cdot\text{min})$ for CBF and $+0.0030\text{ s}$ for ATT, validating that the improvement occurs broadly across test cases.

---

## 6. Statistical Significance vs. Practical / Clinical Significance

A central scientific contribution of this report is clearly distinguishing mathematical significance from practical utility:

1. **Astronomical Statistical Significance:**
   - Because the test cohort consists of $100,000$ synthetic observations, standard errors are extremely small. A $1\text{ ms}$ error reduction yields a paired $t$-statistic of $t = 9.2$ and $p = 5.56 \times 10^{-20}$.
2. **Negligible Practical Effect Size:**
   - In clinical ASL-MRI, scanner repetition times, cardiac pulsations, and bolus transit dispersion introduce variations on the order of $100\text{ to }200\text{ milliseconds}$ ($0.1\text{--}0.2\text{ s}$). An improvement of $0.001\text{ seconds}$ is undetectable in practice.
   - For CBF, an improvement of $0.0417\text{ mL}/(100\text{g}\cdot\text{min})$ represents only $\approx 0.08\%$ of the mean physiological perfusion ($50\text{ mL}/(100\text{g}\cdot\text{min})$).
3. **Synthetic vs. Clinical Generalization:**
   - Evaluating on synthetic Rician noise samples tests estimator consistency under idealized Buxton kinetics. It does **not** demonstrate clinical validity, robustness to in-vivo motion artifacts, vessel crushing deficiencies, or macrovascular transit delays.

---

## 7. Fundamental Identifiability Limitation: The Early-ATT Problem

Both models suffer from elevated error ($\approx 0.47\text{ s}$) in the early arrival range ($[0.5, 1.0)\text{ s}$), where the Feature ELM slightly worsens performance ($-0.64\%$).

### Mathematical & Physical Cause:
In the standard 4-PLD acquisition, the observation times are:
$$\text{PLDs} = [1.525, 2.025, 2.525, 3.025]\text{ s}$$
For tissue with $\text{ATT} < 1.0\text{ s}$, the labeled bolus begins arriving at $t = \text{ATT}$ and completes inflow at $t = \text{ATT} + \tau = \text{ATT} + 1.8\text{ s} \le 2.8\text{ s}$. Crucially, the signal before $1.525\text{ s}$—which contains the inflow slope $\frac{\partial \Delta M}{\partial t} \propto \frac{1}{\text{ATT}}$—is never sampled.

By the time the scanner acquires the first image at $1.525\text{ s}$, the signal has already entered the post-inflow mono-exponential decay phase. At this point, the mathematical derivative with respect to ATT becomes:
$$\frac{\partial \Delta M}{\partial \delta} \approx -\frac{1}{T_{1\text{a}}} \Delta M$$
which is strictly collinear with the CBF scaling parameter $f$. Consequently, early ATT is mathematically unidentifiable under late PLDs. Downstream readouts regularize this uncertainty by shrinking estimates toward the prior distribution mean ($1.75\text{ s}$), inevitably degrading early-ATT accuracy.

---

## 8. Exact Reproduction Instructions

To reproduce all figures and tables from scratch using the existing dataset arrays:

```bash
# 1. Open shell in Drive D workspace
cd "D:\dnn vs dnn elm"

# 2. Execute reproduction script
python results/BEST_RESULTS_PRESENTATION/reproduce_all.py
```

The script executes in $\approx 45\text{ seconds}$ on an NVIDIA GTX 1650 GPU, requires $< 100\text{ MB}$ RAM, and regenerates all CSVs and 300-DPI PNGs into `results/BEST_RESULTS_PRESENTATION/`.

---

## 9. Final Conclusion

1. **Does the Ensemble Feature ELM improve CBF?**  
   **Yes.** It provides a consistent $+1.12\%$ RMSE improvement ($0.0417\text{ mL}/(100\text{g}\cdot\text{min})$) and effectively removes baseline negative bias.
2. **Does it meaningfully improve ATT?**  
   **No.** The ATT improvement is $+0.31\%$ ($1\text{ millisecond}$), which is clinically negligible.
3. **Does it resolve early-ATT transit estimation?**  
   **No.** Early-ATT remains fundamentally unidentifiable under late PLDs ($\ge 1.525\text{ s}$), and regularized readouts slightly degrade early-ATT error by $-0.64\%$.
4. **Is the Feature ELM justified as the project benchmark?**  
   **Yes.** Because it trains in $15\text{ seconds}$ without iterative gradient descent and provides the best overall CBF/ATT trade-off without adding architectural complexity, the **5-Seed Ensemble Feature ELM** is firmly recommended as the project's definitive DNN+ELM estimator.
