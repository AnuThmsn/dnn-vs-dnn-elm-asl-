# Stage 1: Forward Model and Physical Reference Values Audit

**Experiment:** ASL-MRI DNN-ELM Improvement Campaign — Experiment 3: Reference Values, Numerical Scaling, Calibration, and ELM Correction Under Fixed PLDs  
**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Workspace:** `D:\dnn vs dnn elm`  
**Experiment Directory:** `experiments/reference_calibration_03/`  
**Date:** October 13, 2026  
**Status:** Stage 1 Audit Complete  

---

## 1. Executive Audit Overview

The goal of Experiment 3 is to investigate whether CBF and ATT estimation can be improved by:
1. Auditing physical reference values and forward model conventions.
2. Evaluating alternative numerical scaling and target transformations.
3. Conducting sensitivity analyses on physical reference parameters.
4. Implementing post-hoc calibration and shrinkage for DNN-ELM predictions.

The acquisition protocol remains **strictly fixed** at the clinical four-PLD baseline:
$$\text{PLDs} = [1.525, 2.025, 2.525, 3.025]\text{ s}, \quad \tau = 1.8\text{ s}$$

---

## 2. Inventory of Current Physical Reference Constants & Sources

| Parameter | Symbol | Value | Units | Documented Source | Role in Pipeline |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Post-Labeling Delays** | $\text{PLD}$ | $[1.525, 2.025, 2.525, 3.025]$ | s | Ishida et al. (2024); Woods et al. (2020) | Sampling times for bolus inflow & washout |
| **Labeling Duration** | $\tau$ | $1.8$ | s | Alsop et al. (2015 consensus) | Duration of pCASL RF labeling pulse train |
| **Blood Longitudinal Relaxation** | $T_{1a}$ | $1.66$ | s ($1660$ ms) | Lu et al. (2004); Alsop et al. (2015 consensus: $1.65$ s) | Decay of labeled blood during arterial transit |
| **Tissue Longitudinal Relaxation** | $T_{1t}$ | $1.20$ | s ($1200$ ms) | Buxton et al. (1998); Wansapura et al. (1999) | Gray matter tissue $T_1$ decay after capillary exchange |
| **Labeling Efficiency** | $\alpha$ | $0.85$ ($85\%$) | dimensionless | Alsop et al. (2015 consensus recommendation) | Inversion efficiency for pCASL |
| **Transit Attenuation Factor** | $\beta$ | $0.75$ | dimensionless | Ishida et al. (2024) / historical notebook constant | Vascular crushing / microvascular attenuation |
| **Partition Coefficient** | $\lambda$ | $0.90$ | mL/g | Herscovitch & Raichle (1985); Alsop et al. (2015) | Brain blood-to-tissue water partition ratio |
| **Global Signal Multiplier** | $\text{SCALE}$ | $100,000.0$ | arbitrary units | Historical notebook scaling factor | Maps dimensionless signal fractions to large integer range |
| **Reference Perfusion** | $\text{CBF}_{\text{ref}}$ | $50.0$ | mL/(100g$\cdot$min) | Physiological reference point | Midpoint of physiological CBF distribution |
| **Reference Transit Time** | $\text{ATT}_{\text{ref}}$ | $1.60$ | s | Physiological reference point | Typical gray matter transit time at 3T |
| **Reference Signal Scaled** | $\text{SIG\_REF\_SC}$ | $334.2039$ | arbitrary units | Derived at $\text{PLD}=2.0, \tau=1.8, \text{CBF}=50, \text{ATT}=1.6$ | Reference signal magnitude for noise scaling |

---

## 3. Mathematical Audit of Forward Model Implementation

### A. Buxton General Kinetic Model (GKM) Equation
In `experiments/physics.py` and `asl-dnn-papermatched-v2-noise-density.ipynb` (Cell 4), the forward signal is computed as:
$$\Delta M(w) = \text{coeff} \cdot \left[ e^{-\max(w - \delta, 0)/T_{1t}} - e^{-\max(\tau + w - \delta, 0)/T_{1t}} \right]$$
where:
$$\text{coeff} = 2.0 \cdot \alpha \cdot \beta \cdot T_{1t} \cdot \left(\frac{1}{\lambda}\right) \cdot f \cdot e^{-\delta / T_{1a}}$$
and the physiological perfusion rate $f$ is computed from CBF via:
$$f = \frac{\text{CBF}}{6000.0 \cdot \lambda}$$

### B. Trace of the Partition Coefficient $\lambda$
In standard ASL formulation (Alsop et al. 2015):
$$f = \frac{\text{CBF}}{6000.0 \cdot \lambda} \quad \text{or} \quad \Delta M \propto \frac{2 \alpha \beta T_{1t} \cdot \text{CBF}}{6000.0 \cdot \lambda}$$
In the notebook's implementation:
$$\text{coeff} = 2 \alpha \beta T_{1t} \left(\frac{1}{\lambda}\right) \left(\frac{\text{CBF}}{6000.0 \cdot \lambda}\right) e^{-\delta / T_{1a}} = \frac{2 \alpha \beta T_{1t}}{6000.0 \cdot \lambda^2} \text{CBF} \cdot e^{-\delta / T_{1a}}$$
Notice that the factor $(1/\lambda)$ appears twice, producing $(1/\lambda^2) = 1/(0.9)^2 \approx 1.2346$ instead of $1/\lambda = 1/0.9 \approx 1.1111$.
**Pipeline Consistency Check:**
- Is this implementation consistent across simulation, training, validation, and testing? **YES.**
- Because the exact same formula was used in generating `X_train.npy`, `X_val.npy`, `X_test.npy`, and `test_predictions.npz`, the neural network and ELM were trained on this exact mapping. The extra factor of $(1/\lambda)$ is purely an amplitude scaling factor (identical to multiplying `SCALE` by $1/\lambda$).
- Changing this factor post-hoc would arbitrarily scale input signals relative to the frozen neural network's trained weights.

### C. Reference $M_0$ Assumptions
In clinical scanners, $M_0$ is the equilibrium magnetization of arterial blood (or tissue divided by $\lambda$) measured from a proton-density calibration scan.
In this synthetic benchmark:
- $M_0$ is normalized to $1.0$.
- The global multiplier `SCALE = 100_000.0` acts as the effective $M_0$ gain.
- All noise levels and signal magnitudes scale proportionally with `SCALE`.

---

## 4. Target and Input Standardization Audit

### A. Target Normalization
- **CBF:** Target uniform in $[0.0, 100.0]\text{ mL/(100g}\cdot\text{min})$.
  - $\mu_{\text{CBF}} = 50.00099$
  - $\sigma_{\text{CBF}} = 28.86714$ (Matches theoretical $100/\sqrt{12} \approx 28.8675$)
- **ATT:** Target uniform in $[0.5, 3.0]\text{ seconds}$.
  - $\mu_{\text{ATT}} = 1.75155$
  - $\sigma_{\text{ATT}} = 0.72207$ (Matches theoretical $2.5/\sqrt{12} \approx 0.72169$)
- **Inverse Transformation & Bounding Clamps:**
  $$\hat{y}_{\text{physical}} = \operatorname{clamp}\left( \hat{y}_{\text{std}} \cdot \sigma_y + \mu_y, \ y_{\min}, \ y_{\max} \right)$$
  - CBF clamped strictly to $[0.0, 100.0]$.
  - ATT clamped strictly to $[0.5, 3.0]$.

### B. Input Normalization
Observed signals $X \in \mathbb{R}^4$ are standardized channel-wise using statistics fitted strictly on `X_train.npy`:
- $X_{\text{mean}} = [656.0311, \ 560.5862, \ 435.96695, \ 308.37015]$
- $X_{\text{std}} = [480.29407, \ 344.71332, \ 253.01886, \ 179.04381]$
- Transformation:
  $$X_{\text{norm}} = \frac{X - X_{\text{mean}}}{X_{\text{std}} + 10^{-8}}$$

---

## 5. Noise Model & SNR Protocol Audit

### A. Dual-Channel Quadrature Rician Noise
Observed signal $X$ is generated from clean signal $S$ by adding independent Gaussian noise across four channels:
$$M_c = \sqrt{(S + e_1)^2 + e_2^2}, \quad M_l = \sqrt{(S + e_3)^2 + e_4^2}, \quad X = M_c + M_l$$
where $e_i \sim \mathcal{N}(0, \sigma^2)$.
- **Noise Standard Deviation:**
  $$\sigma = \frac{\text{SIG\_REF\_SC}}{\text{SNR}}, \quad \text{SIG\_REF\_SC} \approx 334.2039$$
- **Zero-Noise Limit ($\text{SNR} = \infty$):**
  $$X = S + S = 2 \cdot S$$
- **Training Sweep:** 100 uniformly spaced noise standard deviations in $[0, \text{SD\_MAX}]$, where $\text{SD\_MAX} = \text{SIG\_REF\_SC} / 5.0 \approx 66.8408$.
- **Test Evaluation Checkpoints:** 12 discrete SNR levels: $[5, 8, 10, 12, 15, 20, 30, 50, 100, 200, 500, \infty]$.

---

## 6. Stage 1 Audit Conclusion

1. **Self-Consistency Confirmed:** The forward model equations, parameter values, target ranges, normalization statistics, and noise generators are applied with 100% mathematical consistency across training, validation, and testing.
2. **Frozen Baseline Alignment:** The saved Baseline A checkpoints (`CBFnet_G_baselineA.pth` and `ATTnet_G_baselineA.pth`) match these exact normalization statistics.
3. **Stage 1 Complete:** The configuration has been recorded in `config.json` and this audit is frozen. We proceed to Stage 2: Numerical Scaling and Target Transformations.
