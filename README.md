# Multi-PLD ASL Perfusion Parameter Estimation: DNN vs. DNN-ELM

Investigation and benchmark of Deep Neural Networks (DNN) versus Extreme Learning Machines (ELM) and hybrid ensembles for inverse parameter estimation of **Cerebral Blood Flow (CBF)** and **Arterial Transit Time (ATT)** in multi-Post-Labeling Delay (multi-PLD) Arterial Spin Labeling (ASL) MRI.

---

## 1. Research Overview

Multi-PLD pseudo-continuous Arterial Spin Labeling (pCASL) measures perfusion difference signals across multiple acquisition delay times ($w$). Because ASL signals suffer from low SNR and non-Gaussian Rician magnitude measurement noise, inverting the Buxton General Kinetic Model (GKM) is an ill-posed nonlinear inverse problem.

This project investigates:
1. **DNN Baselines**:
   - **Baseline A (Experiment G)**: Decoupled ELU MLPs (CBFnet: 5 layers $\times$ 128 neurons; ATTnet: 4 layers $\times$ 64 neurons).
   - **Baseline B (Paper-Matched)**: Replicating Ishida et al. (2024), *NMR in Biomedicine* (CBFnet: 9 layers $\times$ 50 neurons; ATTnet: 9 layers $\times$ 100 neurons, MAE loss).
2. **ELM & Hybrid Architectures**:
   - Standalone closed-form regularized Extreme Learning Machines.
   - Weighted linear ensembles ($\alpha \cdot \text{DNN} + (1-\alpha) \cdot \text{ELM}$).
   - Out-of-Fold (OOF) 5-fold cross-validated Residual ELM error compensation.
   - DNN-Feature $\to$ ELM (penultimate latent feature extraction with an analytical ridge head).

---

## 2. Acquisition & Forward Model Settings (Strictly Frozen)

- **Post-Labeling Delays (PLDs)**: $[1.525, 2.025, 2.525, 3.025]\text{ s}$ ($\text{Input Dim} = 4$)
- **Labeling Duration ($\tau$)**: $1.8\text{ s}$
- **Tissue Relaxation ($T_{1t}$)**: $1.2\text{ s}$
- **Blood Relaxation ($T_{1a}$)**: $1.66\text{ s}$
- **Labeling Efficiency ($\alpha$)**: $0.85$
- **Attenuation Factor ($\beta$)**: $0.75$
- **Partition Coefficient ($\lambda$)**: $0.9\text{ mL/g}$
- **Signal Scaling Constant**: $\text{SCALE} = 100,000.0$
- **Reference Signal**: $\text{SIG\_REF\_SC} \approx 334.2039$
- **Physiological Bounds**:
  - $\text{CBF} \in [0.0, 100.0]\ \text{mL} / (100\text{g}\cdot\min)$
  - $\text{ATT} \in [0.5, 3.0]\ \text{s}$
- **Noise Model**: Dual-channel quadrature Rician magnitude noise ($M_c = \sqrt{(S+e_1)^2 + e_2^2}$, $M_l = \sqrt{(S+e_3)^2 + e_4^2}$, $X = M_c + M_l$). Clean limit: $X_{\text{clean}} = 2 \times S$.

---

## 3. Repository Structure

```
.
├── asl-dnn-papermatched-v2-noise-density.ipynb  # Ishida et al. 2024 paper-matched 9-layer MLP
├── dnn-elm-noise (7).ipynb                      # Experiment H: Standalone ELM, ensembles, and hybrids
├── experiments/
│   ├── config_frozen_physics.json               # Machine-readable frozen parameters & protocols
│   ├── physics.py                               # Buxton kinetic model & Rician noise simulator
│   ├── dataset.py                               # Memory-mapped, low-RAM chunked data generation
│   ├── memory_config.py                         # Strict Drive D cache & zero-C-drive-footprint controls
│   ├── train_baseline_A.py                      # Baseline A training, evaluation, and checkpoint pipeline
│   └── cache/                                   # Deterministic cached datasets (seed 42/99/7)
├── audit_notebooks.py                           # Audit parser for historical notebook outputs
└── clean_c_drive.py                             # System utility for clearing OS cache files
```

---

## 4. Quickstart

### Environment Setup
```bash
python -m pip install torch numpy scipy pandas psutil
```

### Reproduce Data Splits & Validate Physics
```bash
python experiments/dataset.py
```

### Train Baseline A (DNN-G)
```bash
python experiments/train_baseline_A.py
```
