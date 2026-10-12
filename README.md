# Multi-PLD ASL Perfusion Parameter Estimation: DNN vs. DNN-ELM

Comprehensive benchmark of Deep Neural Networks (DNN) versus Extreme Learning Machines (ELM) and hybrid ensembles for inverse parameter estimation of **Cerebral Blood Flow (CBF)** and **Arterial Transit Time (ATT)** in multi-Post-Labeling Delay (multi-PLD) Arterial Spin Labeling (ASL) MRI.

---

## 1. Research Overview & Problem Formulation

Multi-PLD pseudo-continuous Arterial Spin Labeling (pCASL) measures perfusion difference signals across four acquisition delay times ($w \in \{1.525, 2.025, 2.525, 3.025\}\text{ s}$). Because ASL signals suffer from low SNR and non-Gaussian Rician magnitude measurement noise, inverting the Buxton General Kinetic Model (GKM) is an ill-posed nonlinear inverse problem.

This study systematically compares:
1. **Frozen DNN Baseline (Baseline A / Experiment G DNN-G)**:
   - CBFnet: 5 hidden layers $\times$ 128 neurons (ELU MLP, 66,817 trainable parameters).
   - ATTnet: 4 hidden layers $\times$ 64 neurons (ELU MLP, 12,865 trainable parameters).
   - Loss: Standardized MSE Loss.
2. **Standalone Regularized ELM**:
   - Closed-form analytical solve $\beta = (H^T H + \lambda I)^{-1} H^T Y$.
   - Hyperparameters tuned strictly on the validation set ($H=1024, \lambda=10^{-4}$).
3. **Weighted DNN + ELM Ensemble**:
   - $\hat{Y}_{\text{ens}} = \alpha \hat{Y}_{\text{DNN}} + (1 - \alpha) \hat{Y}_{\text{ELM}}$ ($\alpha_{\text{CBF}}=0.65, \alpha_{\text{ATT}}=0.95$ tuned on validation).
4. **Residual ELM Correction**:
   - Direct learning of systematic residual errors from input signals and DNN predictions with validation-calibrated shrinkage.
5. **DNN-Feature ELM & 5-Seed Ensemble Feature ELM**:
   - Freezes the DNN backbone and extracts 128-dimensional (CBF) and 64-dimensional (ATT) penultimate layer representations.
   - Solves an analytical ridge readout, eliminating gradient optimization instability and achieving state-of-the-art performance.

---

## 2. Key Experimental Findings (Locked Test Set: $N = 100,000$ Samples)

All models evaluated on identical test samples with dual-channel Rician noise across 12 SNR levels ($5 \le \text{SNR} \le \infty$):

| Model Architecture | CBF RMSE $\downarrow$ | CBF MAE $\downarrow$ | CBF CCC $\uparrow$ | ATT RMSE $\downarrow$ | ATT MAE $\downarrow$ | ATT CCC $\uparrow$ | Paired Bootstrap $p$-value |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline A (DNN-G)** | 3.7266 | 2.5323 | 0.9916 | 0.3058 | 0.1935 | 0.9009 | *Reference* |
| **Standalone ELM (1024)** | 3.7478 | 2.5457 | 0.9915 | 0.3162 | 0.2061 | 0.8933 | $p < 10^{-8}$ (Inferior) |
| **Weighted Ensemble** | 3.7147 | 2.5176 | 0.9916 | 0.3058 | 0.1938 | 0.9008 | $p < 10^{-15}$ (CBF only) |
| **Residual ELM** | 3.6836 | 2.4395 | 0.9918 | 0.3054 | 0.1912 | 0.9012 | $p < 10^{-12}$ (Superior) |
| **DNN-Feature ELM** | 3.6851 | 2.4248 | 0.9918 | 0.3050 | 0.1906 | 0.9015 | $p < 10^{-9}$ (Superior) |
| **Ensemble Feature ELM (5 seeds)** | **3.6828** | **2.4232** | **0.9918** | **0.3049** | **0.1904** | **0.9016** | **$p = 2.45 \times 10^{-14}$ (Best Overall)** |

### Clean Population Performance ($N = 2,392$ Noiseless Points)
- **Baseline A (DNN-G)**: CBF RMSE = **2.6979**, MAE = 1.7397 | ATT RMSE = **0.1951**, MAE = 0.1175
- **DNN-Feature ELM**: CBF RMSE = **2.6153** (**+3.06% gain**, MAE **+15.46% gain**) | ATT RMSE = **0.1942** (**+0.48% gain**, MAE **+4.59% gain**)
- **Ensemble Feature ELM**: CBF RMSE = **2.6163** (**+3.03% gain**) | ATT RMSE = **0.1940** (**+0.56% gain**, MAE **+5.09% gain**)

---

## 3. Statistical Significance (1,000 Paired Bootstrap Resamples)

- **CBF Gain (Ensemble Feature ELM vs DNN)**:
  - $\Delta \text{RMSE} = +0.0438\ \text{mL}/(100\text{g}\cdot\min)$ with $95\%\ \text{CI}\ [0.0396, 0.0484]$ (**+1.18% gain**, $p < 10^{-15}$).
  - $\Delta \text{MAE} = +0.1091\ \text{mL}/(100\text{g}\cdot\min)$ with $95\%\ \text{CI}\ [0.1058, 0.1124]$ (**+4.31% gain**).
- **ATT Gain (Ensemble Feature ELM vs DNN)**:
  - $\Delta \text{RMSE} = +0.00093\ \text{s}$ with $95\%\ \text{CI}\ [0.00069, 0.00117]$ (**+0.30% gain**, $p = 2.45 \times 10^{-14}$).
  - $\Delta \text{MAE} = +0.00314\ \text{s}$ with $95\%\ \text{CI}\ [0.00299, 0.00329]$ (**+1.62% gain**).

---

## 4. Repository Structure

```
.
├── experiments/
│   ├── baseline_A/                              # Frozen Baseline A (DNN-G)
│   │   ├── ATTnet_G_baselineA.pth               # 4x64 ELU MLP checkpoint
│   │   ├── CBFnet_G_baselineA.pth               # 5x128 ELU MLP checkpoint
│   │   ├── normalization_stats.npz              # Scaler parameters (Train only)
│   │   ├── training_history.json                # Complete epoch loss traces
│   │   └── baselineA_test_predictions.npz       # Raw and physical test outputs
│   ├── elm_models/                              # ELM & Hybrid implementations
│   │   ├── elm_engine.py                        # Low-RAM chunked GPU ELM solver
│   │   ├── run_elm_comparisons.py               # Systematic comparison pipeline
│   │   ├── statistical_significance.py          # Paired bootstrap significance testing
│   │   ├── all_models_test_comparison.csv       # Test set metric tables
│   │   ├── all_models_clean_comparison.csv      # Clean grid metric tables
│   │   └── all_models_snr_comparison.csv        # 12-level SNR robustness breakdown
│   ├── diagnostics/                             # Verified plots & diagnostics
│   │   ├── training_curves_CBF.png              # Verified CBF convergence
│   │   ├── training_curves_ATT.png              # Verified ATT convergence
│   │   ├── predicted_vs_true_scatter.png        # 1:1 Identity scatter plots
│   │   ├── residual_diagnostics.png             # Error distributions & heteroscedasticity
│   │   ├── all_models_rmse_comparison.png       # Head-to-head bar chart
│   │   └── all_models_snr_curves.png            # SNR robustness profiles
│   ├── dataset.py                               # Low-RAM memory-mapped generator
│   └── physics.py                               # Buxton kinetic forward simulator
└── README.md
```

---

## 5. Reproduction Instructions

```bash
# 1. Verify Baseline Integrity & Artifact Hashes
python experiments/verify_baseline_integrity.py

# 2. Generate Diagnostics & Training Curves
python experiments/generate_diagnostics.py

# 3. Execute Full ELM Experimental Campaign
python experiments/elm_models/run_elm_comparisons.py

# 4. Run Paired Bootstrap Significance Tests
python experiments/elm_models/statistical_significance.py
```
