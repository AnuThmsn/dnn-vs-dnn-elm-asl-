# Stage 3: Physical Reference Value Sensitivity and Robustness Report

**Experiment:** ASL-MRI DNN-ELM Improvement Campaign — Experiment 3: Reference Values, Numerical Scaling, Calibration, and ELM Correction Under Fixed PLDs  
**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Workspace:** `D:\dnn vs dnn elm`  
**Experiment Directory:** `experiments/reference_calibration_03/`  
**Date:** October 13, 2026  
**Status:** Stage 3 Sensitivity Analysis Complete  

---

## 1. Executive Summary & Physics Rationale

In clinical ASL MRI, biological parameters in vivo deviate from nominal literature reference values due to physiological heterogeneity (hematocrit, age, sex, tissue composition, and vascular health). This stage performed a **controlled sensitivity analysis** to quantify how estimator errors (Baseline A DNN and Feature ELM) behave when actual physical parameters vary across scientifically defensible ranges:

1. **Blood Relaxation Time ($T_{1a}$):** $[1.50, 1.80]\text{ s}$ (Lu et al. 2004; Alsop et al. 2015 consensus: $1.65 \pm 0.10\text{ s}$).
2. **Tissue Relaxation Time ($T_{1t}$):** $[1.10, 1.40]\text{ s}$ (Buxton et al. 1998; Wansapura et al. 1999: $1.10 - 1.40\text{ s}$ for GM at 3T).
3. **Inversion Efficiency ($\alpha$):** $[0.75, 0.90]$ (Alsop et al. 2015 consensus: nominal $0.85$, degraded by carotid velocity/off-resonance).
4. **Blood-Tissue Partition Coefficient ($\lambda$):** $[0.82, 0.98]\text{ mL/g}$ (Herscovitch & Raichle 1985: $0.82$ WM, $0.98$ GM, $0.90$ whole brain).
5. **Signal Scale Multiplier ($M_0$):** $[0.80, 1.20] \times \text{SCALE}$ (Accounting for $\pm 20\%$ B1 receive coil sensitivity / M0 calibration drift).

---

## 2. Quantitative Sensitivity Results ($N=50,000$ Held-Out Validation)

| Parameter | Tested Value | Status | DNN CBF RMSE | ELM CBF RMSE | DNN CBF Bias | ELM CBF Bias | DNN ATT RMSE (s) | ELM ATT RMSE (s) | DNN ATT Bias (s) | ELM ATT Bias (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Blood $T_{1a}$ (s)** | 1.50 | -9.6% | 7.558 | 7.269 | -5.40 | -5.18 | 0.3038 | 0.3033 | -0.020 | -0.015 |
| | 1.58 | -4.8% | 4.953 | 4.750 | -2.71 | -2.51 | 0.3022 | 0.3018 | -0.014 | -0.010 |
| | **1.66** | **Baseline** | **3.741** | **3.712** | **-0.17** | **+0.02** | **0.3011** | **0.3008** | **-0.008** | **-0.004** |
| | 1.74 | +4.8% | 4.647 | 4.761 | +2.12 | +2.29 | 0.3007 | 0.3006 | -0.003 | +0.001 |
| | 1.80 | +8.4% | 5.968 | 6.114 | +3.65 | +3.79 | 0.3008 | 0.3009 | +0.001 | +0.005 |
| **Tissue $T_{1t}$ (s)** | 1.10 | -8.3% | 6.198 | 6.192 | -3.96 | -3.91 | 0.3147 | 0.3133 | -0.053 | -0.049 |
| | 1.15 | -4.2% | 4.550 | 4.493 | -1.98 | -1.87 | 0.3058 | 0.3051 | -0.033 | -0.028 |
| | **1.20** | **Baseline** | **3.741** | **3.712** | **-0.17** | **+0.02** | **0.3011** | **0.3008** | **-0.008** | **-0.004** |
| | 1.25 | +4.2% | 3.954 | 4.033 | +1.44 | +1.67 | 0.3079 | 0.3075 | +0.027 | +0.029 |
| | 1.30 | +8.3% | 4.850 | 5.010 | +2.96 | +3.19 | 0.3173 | 0.3166 | +0.050 | +0.051 |
| | 1.40 | +16.7% | 7.479 | 7.664 | +5.86 | +6.02 | 0.3447 | 0.3435 | +0.100 | +0.100 |
| **Alpha ($\alpha$)** | 0.75 | -11.8% | 7.714 | 7.503 | -5.92 | -5.73 | 0.3044 | 0.3040 | -0.019 | -0.014 |
| | 0.80 | -5.9% | 5.066 | 4.903 | -3.00 | -2.81 | 0.3024 | 0.3021 | -0.015 | -0.010 |
| | **0.85** | **Baseline** | **3.741** | **3.712** | **-0.17** | **+0.02** | **0.3011** | **0.3008** | **-0.008** | **-0.004** |
| | 0.90 | +5.9% | 4.690 | 4.795 | +2.50 | +2.67 | 0.3020 | 0.3020 | -0.001 | +0.004 |
| **Lambda ($\lambda$)** | 0.82 (WM) | -8.9% | 10.019 | 10.138 | +8.21 | +8.24 | 0.3148 | 0.3155 | +0.020 | +0.023 |
| | 0.86 | -4.4% | 5.912 | 6.046 | +4.04 | +4.19 | 0.3040 | 0.3041 | +0.005 | +0.009 |
| | **0.90** | **Baseline** | **3.741** | **3.712** | **-0.17** | **+0.02** | **0.3011** | **0.3008** | **-0.008** | **-0.004** |
| | 0.94 | +4.4% | 6.080 | 5.892 | -4.21 | -4.02 | 0.3032 | 0.3029 | -0.017 | -0.012 |
| | 0.98 (GM) | +8.9% | 9.733 | 9.505 | -7.88 | -7.69 | 0.3053 | 0.3049 | -0.021 | -0.016 |
| **Scale / $M_0$** | 80,000 | -20.0% | 12.095 | 11.854 | -10.08 | -9.89 | 0.3059 | 0.3054 | -0.022 | -0.017 |
| | 90,000 | -10.0% | 6.852 | 6.651 | -5.04 | -4.85 | 0.3038 | 0.3034 | -0.018 | -0.013 |
| | **100,000** | **Baseline** | **3.741** | **3.712** | **-0.17** | **+0.02** | **0.3011** | **0.3008** | **-0.008** | **-0.004** |
| | 110,000 | +10.0% | 6.088 | 6.223 | +4.24 | +4.38 | 0.3043 | 0.3045 | +0.005 | +0.009 |
| | 120,000 | +20.0% | 9.848 | 9.969 | +8.04 | +8.09 | 0.3142 | 0.3149 | +0.019 | +0.023 |

---

## 3. Scientific Discoveries & Physical Explanations

### Discovery 1: CBF is Dominated by Global Linear Scale Mismatches
- **Scale / $M_0$ Drift:** A $\pm 10\%$ calibration drift in signal scale directly creates a $\mp 5\text{ mL/100g/min}$ bias in CBF ($\approx 10\%$ of mean CBF), doubling CBF RMSE from $3.74$ to $6.85\text{ mL/100g/min}$.
- **Inversion Efficiency ($\alpha$):** A drop from $0.85 \to 0.75$ (common with B0/B1 off-resonance in carotids) produces a severe underestimation of CBF (Bias: $-5.92\text{ mL/100g/min}$, RMSE: $7.71$).
- **Partition Coefficient ($\lambda$):** Because $(1/\lambda)^2$ enters the simulator, evaluating white matter ($\lambda=0.82$) with whole-brain assumptions causes a large overestimation bias ($+8.21\text{ mL/100g/min}$).

### Discovery 2: ATT is Exceptionally Robust to Amplitude Scale
- In stark contrast to CBF, **ATT estimation is virtually immune to linear amplitude errors** ($\alpha$ or $M_0$ scaling):
  - When signal scale drops by $20\%$ ($80,000$ vs $100,000$), ATT RMSE shifts by only $+0.0048\text{ s}$ ($0.3011 \to 0.3059\text{ s}$), and ATT bias shifts by only $-0.014\text{ s}$.
  - When $\alpha$ drops by $11.8\%$ ($0.75$ vs $0.85$), ATT RMSE shifts by only $+0.0033\text{ s}$.
- **Reason:** Transit delay is encoded in the temporal decay shape and relative ratios across the four PLDs ($[1.525, 2.025, 2.525, 3.025]\text{ s}$), not in the overall signal height.

### Discovery 3: Tissue $T_{1t}$ is the Governing Parameter for ATT
- When tissue $T_{1t}$ deviates from $1.20\text{ s}$:
  - At $T_{1t} = 1.40\text{ s}$ (slow tissue decay), the four PLDs exhibit shallower temporal decay slopes. The estimator mistakes this slow washout for delayed arterial arrival, introducing a large positive ATT bias of **$+0.100\text{ s}$** and increasing ATT RMSE to **$0.3447\text{ s}$** ($+14.5\%$ error increase).
  - At $T_{1t} = 1.10\text{ s}$ (fast tissue decay), ATT bias shifts to **$-0.053\text{ s}$**.
- Therefore, **tissue $T_1$ uncertainty represents the primary physical source of ATT estimation error**, whereas blood $T_1$ and $M_0$ scale primarily affect CBF.

---

## 4. Stage 3 Conclusion & Next Step

1. **Distinction Established:**
   - Errors in $\alpha$, $\lambda$, and $M_0$ create pure additive/multiplicative **calibration bias** in CBF without altering ATT.
   - Errors in tissue $T_{1t}$ alter the **temporal dynamics**, causing genuine shape distortion in ATT.
2. **Implications for Stage 4 (Calibration):**
   - The finding that scale mismatches cause pure linear bias in CBF strongly motivates Stage 4: testing linear calibration and shrinkage mappings on DNN-ELM predictions to rectify residual calibration offsets.
3. **Stage 3 Completed:** We commit Stage 3 files before proceeding to Stage 4.
