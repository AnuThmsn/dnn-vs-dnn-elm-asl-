# Experiment 4 Final Report: Multi-Layer Feature ELM for ASL-MRI Parameter Estimation

**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Working Directory:** `D:\dnn vs dnn elm\experiments\multilayer_elm_04`  
**Execution Date:** 2026-10-13  
**Acquisition Protocol (Strictly Frozen):**
- PLDs: $[1.525, 2.025, 2.525, 3.025]$ seconds, Label Duration $\tau = 1.8$ seconds
- Forward Model: Buxton General Kinetic Model (pCASL), Dual-channel Rician noise
- Tissue & Blood Parameters: $T_{1,\text{blood}} = 1.66\text{ s}$, $T_{1,\text{tissue}} = 1.20\text{ s}$, $\alpha = 0.85$, $\beta = 0.75$, $\lambda = 0.90\text{ mL/g}$
- Base Estimators: Frozen Checkpoints (`CBFnet_G_baselineA.pth`, `ATTnet_G_baselineA.pth`)

---

## 1. Executive Summary & Objective

Experiment 4 investigated whether combining intermediate and penultimate DNN hidden representations (multi-layer representations) or employing alternative nonlinear readout mechanisms (ELU, ReLU, Random Fourier Features Kernel-ELM, compact linear ridge regression) improves cerebral blood flow (CBF) and arterial transit time (ATT) parameter estimation over the frozen Baseline A DNN and the historical single-layer Feature ELM.

The investigation was conducted across five sequential, non-destructive stages with strict separation between training/validation ($N_{\text{train}} = 1,000,000$, $N_{\text{val}} = 100,000$) and the locked test cohort ($N_{\text{test}} = 100,000$).

### Key Empirical Findings:
1. **Multi-layer representations do not meaningfully outperform penultimate features:**
   - On the locked test set, the Multi-Layer Feature ELM achieved CBF $\text{RMSE} = 3.6869\text{ mL}/(100\text{g}\cdot\text{min})$ compared to $3.6876\text{ mL}/(100\text{g}\cdot\text{min})$ for the historical penultimate Feature ELM—an absolute difference of only $0.0007\text{ mL}/(100\text{g}\cdot\text{min})$ ($+0.02\%$ relative gain), which is clinically and practically imperceptible.
   - For ATT, both the Multi-Layer Feature ELM and the Historical Penultimate Feature ELM achieved identical locked test accuracy of $\text{RMSE} = 0.3050\text{ s}$ ($0.0000\text{ s}$ difference, $0.00\%$ relative gain).
2. **Alternative nonlinear activations (ELU, ReLU, Kernel-ELM) provide no generalization gain:**
   - Standard Tanh ($\text{RMSE} = 3.7038$ on val) performed equally or better than ELU ($3.7043$) and ReLU ($3.7061$).
   - Random Fourier Feature (RFF) Kernel-ELM was either equivalent ($\gamma = 0.01$, $\text{RMSE} = 3.7124$) or degraded severely ($\gamma \ge 0.1$, $\text{RMSE} = 5.4378$ to $21.3436$) due to high-frequency harmonic overfitting on continuous neural activations.
3. **Early-ATT ($[0.5, 1.0)\text{ s}$) remains fundamentally unimproved:**
   - On the locked test cohort, early-ATT RMSE for the Multi-Layer ELM was $0.4696\text{ s}$ versus $0.4664\text{ s}$ for the base DNN ($+0.0032\text{ s}$ slight error increase). No downstream architectural modifications can overcome the complete absence of early bolus transit dynamics when the earliest PLD is $1.525\text{ s}$.
4. **Statistical significance without practical significance:**
   - While paired statistical tests yield astronomically small $p$-values ($p < 10^{-14}$) due to the sheer statistical power of $N=100,000$ test observations, the actual effect size over the historical Feature ELM is $< 0.02\%$.

---

## 2. Locked Test Evaluation Benchmarks ($N=100,000$)

All candidate models were frozen prior to test-set evaluation and evaluated on the exact same locked test samples:

### Table 1: Overall Parameter Estimation Accuracy on Locked Test Cohort

| Estimator / Model | CBF RMSE | CBF MAE | CBF CCC | CBF Imp vs DNN | CBF 95% Bootstrap CI | Paired $t$-test $p$ | ATT RMSE | ATT MAE | ATT CCC | Early ATT RMSE $[0.5, 1.0)\text{s}$ | ATT Imp vs DNN | ATT 95% Bootstrap CI | Paired $t$-test $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A DNN-G** (Frozen) | 3.7266 | 2.5323 | 0.9916 | 0.00% | $[0.000, 0.000]$ | — | 0.3058 | 0.1935 | 0.9009 | **0.4664** | 0.00% | $[0.0000, 0.0000]$ | — |
| **Historical Feature ELM** (Rep A, 512, $\alpha=10^{-2}$) | 3.6876 | **2.4313** | **0.9918** | +1.05% | $[-0.0432, -0.0351]$ | $5.34 \times 10^{-81}$ | **0.3050** | **0.1907** | **0.9015** | 0.4694 | **+0.28%** | $[-0.00106, -0.00063]$ | $3.53 \times 10^{-14}$ |
| **Penultimate Candidate** (Rep A, 512, $\alpha=1.0$) | 3.6875 | 2.4318 | **0.9918** | +1.05% | $[-0.0433, -0.0353]$ | $4.19 \times 10^{-84}$ | **0.3050** | 0.1908 | **0.9015** | 0.4695 | **+0.28%** | $[-0.00106, -0.00064]$ | $1.19 \times 10^{-15}$ |
| **Multi-Layer Candidate** (Rep C: $[h_1, h_5]$, 512) | **3.6869** | 2.4339 | **0.9918** | **+1.06%** | $[-0.0439, -0.0356]$ | $1.08 \times 10^{-81}$ | **0.3050** | **0.1907** | **0.9015** | 0.4696 | +0.27% | $[-0.00102, -0.00063]$ | $5.07 \times 10^{-17}$ |
| **Compact Linear Ridge** (Rep A, Direct Ridge) | 3.6926 | 2.4458 | 0.9917 | +0.91% | $[-0.0376, -0.0306]$ | $1.56 \times 10^{-83}$ | 0.3053 | 0.1913 | 0.9013 | 0.4705 | +0.15% | $[-0.00059, -0.00033]$ | $1.62 \times 10^{-11}$ |

*Note: 95% bootstrap confidence intervals were computed with $B=2,000$ paired resamples using fixed random seed 42. Paired $t$-test was computed on per-sample squared errors.*

---

## 3. Stratified Evaluation Breakdown

### 3.1 Breakdown Across Physiological ATT Intervals (Locked Test Cohort)

| Bolus Transit Interval | Sample Size | Baseline DNN RMSE | Historical ELM RMSE | Multi-Layer ELM RMSE | Multi-Layer vs DNN $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $[0.5, 1.0)\text{ s}$ (Early Arrival) | 19,728 | **0.4664** | 0.4694 | 0.4696 | $+0.0032\text{ s}$ (Slightly Degraded) |
| $[1.0, 1.5)\text{ s}$ (Normal Cortical GM) | 20,281 | 0.2948 | **0.2947** | 0.2948 | $0.0000\text{ s}$ (Identical) |
| $[1.5, 2.0)\text{ s}$ (Intermediate) | 19,836 | 0.2542 | **0.2517** | **0.2517** | $-0.0025\text{ s}$ (Modest Gain) |
| $[2.0, 2.5)\text{ s}$ (Delayed Transit) | 20,275 | 0.1699 | 0.1661 | **0.1658** | $-0.0041\text{ s}$ (Modest Gain) |
| $[2.5, 3.0]\text{ s}$ (Late Arrival) | 19,880 | 0.2681 | 0.2630 | **0.2628** | $-0.0053\text{ s}$ (Modest Gain) |

### 3.2 Breakdown Across Acquisition SNR Levels (Locked Test Cohort)

| SNR Level | DNN CBF RMSE | Hist ELM CBF RMSE | Multi-Layer CBF RMSE | DNN ATT RMSE | Hist ELM ATT RMSE | Multi-Layer ATT RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.0 dB** | 9.0747 | 9.0560 | 9.0558 | 0.5050 | 0.5042 | 0.5041 |
| **8.0 dB** | 6.1362 | 6.0963 | 6.0954 | 0.4357 | 0.4347 | 0.4347 |
| **10.0 dB** | 4.8812 | 4.8340 | 4.8331 | 0.3888 | 0.3878 | 0.3877 |
| **12.0 dB** | 3.8744 | 3.8247 | 3.8241 | 0.3444 | 0.3435 | 0.3435 |
| **15.0 dB** | 2.8091 | 2.7594 | 2.7590 | 0.2858 | 0.2850 | 0.2850 |
| **20.0 dB** | 1.8398 | 1.7854 | 1.7850 | 0.2109 | 0.2101 | 0.2101 |
| **30.0 dB** | 0.9997 | 0.9419 | 0.9416 | 0.1340 | 0.1332 | 0.1332 |
| **50.0 dB** | 0.6970 | 0.6409 | 0.6406 | 0.1017 | 0.1009 | 0.1009 |
| **100.0 dB** | 0.6471 | 0.5912 | 0.5909 | 0.0968 | 0.0959 | 0.0959 |
| **200.0 dB** | 0.6403 | 0.5843 | 0.5840 | 0.0961 | 0.0953 | 0.0953 |
| **500.0 dB** | 0.6391 | 0.5831 | 0.5828 | 0.0960 | 0.0952 | 0.0952 |
| **Inf dB** | 0.6389 | 0.5829 | 0.5826 | 0.0960 | 0.0952 | 0.0951 |

---

## 4. Explicit Answers to the Seven Required Questions

### 1. Do multi-layer features outperform penultimate features?
**No.**  
Across all experiments (Stage 2 validation sweep and Stage 5 locked test cohort):
- For **CBF**, penultimate features (Rep A, 128 units) achieved locked test $\text{RMSE} = 3.6875\text{ mL}/(100\text{g}\cdot\text{min})$. Concatenating an early layer ($[h_1, h_5]$, 256 units) yielded $\text{RMSE} = 3.6869$, an improvement of only $0.0006\text{ mL}/(100\text{g}\cdot\text{min})$ ($+0.016\%$), while MAE slightly worsened from $2.4318$ to $2.4339$. Adding intermediate layers ($[h_4, h_5]$, $[h_1, h_3, h_5]$) consistently degraded validation performance ($3.7059$ to $3.7107$ vs $3.7021$ for penultimate alone).
- For **ATT**, the Multi-Layer candidate ($[h_1, h_4]$) and the Penultimate candidate ($h_4$) achieved **identical** test RMSE to four decimal places ($0.3050\text{ s}$).
Intermediate hidden representations carry earlier, less-task-distilled signal variations that do not add predictive mutual information to the penultimate layer.

### 2. Does a nonlinear readout improve generalization?
**No meaningful improvement over standard Tanh, and Random Fourier Features degrade.**
- In Stage 3, standard $\tanh$ activations performed marginally better than or equivalent to alternative activations (ELU: $3.7043$, ReLU: $3.7061$ vs Tanh: $3.7038$ on validation CBF).
- Kernel-ELM via Random Fourier Features (RFF) with Gaussian RBF kernel failed to provide any benefit: at $\gamma=0.01$, validation RMSE was $3.7124$ (worse than standard ELM $3.7038$); at $\gamma \ge 0.1$, performance collapsed severely ($\text{RMSE} = 5.4378$ to $21.3436$) due to high-frequency spectral oscillations over continuous latent activations.
- Notably, the **Compact Linear Ridge Readout** (direct linear ridge regression without any random hidden units) captured $+0.91\%$ of the total $+1.05\%$ CBF improvement and $+0.15\%$ of the $+0.28\%$ ATT improvement in just $0.8$ seconds of training time.

### 3. Are improvements consistent across seeds and SNR levels?
**Yes, but the effect sizes are universally microscopic.**
- Across 5 independent random initialization seeds ($42, 43, 44, 45, 46$), standard deviation was extremely small: $\sigma_{\text{RMSE}} \approx 0.0006\text{ mL}/(100\text{g}\cdot\text{min})$ for CBF and $\sigma_{\text{RMSE}} \approx 0.00005\text{ s}$ for ATT.
- Across all 12 SNR levels ($5.0\text{ dB}$ to $\infty\text{ dB}$), the multi-layer ELM tracked the historical Feature ELM within $\pm 0.001\text{ mL}/(100\text{g}\cdot\text{min})$ on CBF and $\pm 0.0001\text{ s}$ on ATT.
- The consistency is high because the underlying models are virtually identical in output manifold.

### 4. Does ATT improve without degrading CBF?
**Yes, CBF and ATT maintain their respective small improvements independently, but neither is materially improved.**
- In Experiment 4, CBF and ATT feature extractors and readout models were trained strictly independently.
- CBF RMSE improved by $+1.06\%$ ($3.6869$ vs $3.7266$ for DNN) while ATT RMSE improved by $+0.28\%$ ($0.3050$ vs $0.3058$ for DNN). There is no trade-off or interference between them.
- However, neither parameter achieved a clinically meaningful gain over the historical Feature ELM (CBF gain: $+0.02\%$; ATT gain: $0.00\%$).

### 5. Does early-ATT performance improve meaningfully?
**No. Early-ATT estimation remains completely unimproved and slightly degrades.**
- On the locked test cohort for $\text{ATT} \in [0.5, 1.0)\text{ s}$, baseline DNN RMSE was $0.4664\text{ s}$.
- The Historical Feature ELM achieved $0.4694\text{ s}$, and the Multi-Layer ELM achieved $0.4696\text{ s}$ (both are slightly worse by $+0.003\text{ s}$ due to global regularized shrinkage toward the prior mean of $1.75\text{ s}$).
- This confirms the rigorous physical conclusion established in previous audits: with fixed PLDs at $[1.525, 2.025, 2.525, 3.025]\text{ s}$, no measurement point samples the bolus arrival or tissue accumulation curve before $1.525\text{ s}$. When $t_{\text{arrival}} < 1.0\text{ s}$, the signal has already entered the post-inflow mono-exponential decay phase at all four PLDs. No architectural, multi-layer, or nonlinear readout modification can extract information that does not exist in the physical measurements.

### 6. Is the extra complexity justified?
**No.**  
Concatenating multi-layer intermediate representations doubles to quadruples the input dimensionality (from 128 to 256–640 units), increases memory buffers and matrix sizes, requires additional feature normalization or projection, and doubles training/inference latency, while delivering an absolute improvement of $0.0007\text{ mL}/(100\text{g}\cdot\text{min})$ on CBF ($+0.02\%$) and exactly $0.0000\text{ s}$ on ATT ($0.00\%$). The additional complexity is entirely unjustified.

### 7. If all gains remain negligible, should the project retain the historical ELM as the preferred ELM baseline?
**Yes, conclusively.**  
The Historical Penultimate Feature ELM (Single-layer penultimate representations $h_5$ for CBF, $h_4$ for ATT with 512 hidden units, Tanh activation, and ridge parameter $\alpha \in [10^{-2}, 1.0]$) represents the cleanest, simplest, most reproducible, and computationally lightest ELM architecture. Experiment 1 (hyperparameter tuning), Experiment 2 (physics features/residual learning), Experiment 3 (calibration/shrinkage), and Experiment 4 (multi-layer representations and nonlinear readouts) all demonstrate that downstream mathematical modifications upon the frozen DNN under the fixed 4-PLD protocol yield negligible gains ($< 0.1\%$). The historical Feature ELM should be firmly retained as the definitive ELM benchmark.

---

## 5. Artifact and Verification Inventory

The following artifacts have been generated in `experiments/multilayer_elm_04/`:
- `config.json`: Hardware, architecture, parameter, and layer definitions.
- `STAGE1_ARCHITECTURE_AUDIT.md`: Non-invasive verification report (0.00 max absolute diff).
- `stage2_conditioning_audit.csv`: Numerical condition numbers across representations.
- `stage2_representation_comparison.csv`: Validation sweep comparing Rep A, B, C, D, E.
- `stage3_nonlinear_comparison.csv`: Validation sweep comparing Tanh, ELU, ReLU, RFF, Linear.
- `stage3_seed_variability.csv`: 5-seed evaluation records for candidate models.
- `STAGE4_VALIDATION_REPORT.md`: Comprehensive validation diagnostics report.
- `stage4_validation_comparison.csv`: Summary validation table.
- `stage4_snr_stratified.csv`: Validation SNR breakdowns across 12 levels.
- `stage4_att_intervals.csv`: Validation ATT interval breakdowns.
- `final_test_metrics.csv`: Locked-test metrics ($N=100,000$), paired $t$-tests, bootstrap CIs.
- `final_test_snr_stratified.csv`: Locked-test metrics stratified by 12 SNR levels.
- `final_test_att_intervals.csv`: Locked-test metrics across 5 physiological ATT intervals.
- `plots/stage2_representation_comparison.png`: Representation comparison bar chart.
- `plots/stage4_snr_performance.png`: Validation SNR curves.
- `plots/stage4_att_intervals.png`: Validation ATT interval bar chart.
- `plots/final_test_evaluation.png`: Publication-quality locked test comparison figure.
