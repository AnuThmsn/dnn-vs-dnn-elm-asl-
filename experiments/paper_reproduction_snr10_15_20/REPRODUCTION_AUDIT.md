# REPRODUCTION FORENSIC AUDIT: DNN-G Baseline & Proposed Residual H-DELM Model

**Research Paper:** *A Residual Hybrid DNN–ELM Model for CBF and ATT Estimation in Multi-Delay ASL MRI*  
**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Workspace:** `D:\dnn vs dnn elm`  
**Date:** November 1, 2026  
**Auditor:** Antigravity Autonomous Research Assistant  

---

## 1. Executive Forensic Summary

This forensic audit investigates the local codebase and artifacts to reconstruct the exact implementation for the research paper *A Residual Hybrid DNN–ELM Model for CBF and ATT Estimation in Multi-Delay ASL MRI*.

### Key Verified Code Facts vs. Paper Discrepancies

| Component | Paper Specification | Local Repository Implementation | Status & Provenance |
| :--- | :--- | :--- | :--- |
| **Buxton Forward Model** | Standard pCASL GKM, $\tau=1.8$ s, 4 PLDs | `experiments/physics.py`, `asl_signal()` & `compute_physical_signals()` | **Verified Identical** |
| **Noise Model** | Dual-channel quadrature Rician magnitude ($M_c + M_l$) | `experiments/physics.py`, `add_rician_noise()` | **Verified Identical** |
| **Acquisition PLDs** | $[1.525, 2.025, 2.525, 3.025]$ s | `PLDS = np.array([1.525, 2.025, 2.525, 3.025])` | **Verified Identical** |
| **Physiological Ranges** | $\text{CBF} \in [0, 100]$ mL/100g/min, $\text{ATT} \in [0.5, 3.0]$ s | `physics.CBF_MIN=0, CBF_MAX=100, ATT_MIN=0.5, ATT_MAX=3.0` | **Verified Identical** |
| **DNN-G Baseline Backbone** | 4 hidden layers $\times$ 64 neurons for **BOTH** CBF and ATT | ATT: 4 layers $\times$ 64 neurons.<br>CBF: 5 layers $\times$ 128 neurons (`CBFnet_G_baselineA.pth`). | **Discrepancy in CBFnet:** Existing CBF checkpoint is 128-neuron, 5-layer. A paper-matched 4x64 CBF model must be trained. |
| **DNN-G Output Head** | Bounded Sigmoid transformation: $y_{\min} + (y_{\max} - y_{\min}) \cdot \sigma(z)$ | `StandardizedNet`: linear output on normalized targets, clamped post-prediction. | **Structural Difference:** Paper specifies bounded sigmoid output head. |
| **Physics Features** | Exactly 15 features (4 PLDs, 3 slopes, 3 ratios, mean, var, min, max, trapezoidal AUC) | `feature_engineering.py` contains 10-feature Rep D and 7-feature Rep C. | **Missing Component:** The exact 15-feature extractor combining these 8 groups must be explicitly constructed. |
| **Combined Feature Dim** | $64\ (\text{DNN}) + 15\ (\text{Physics}) = 79$ dimensions | `feature_fusion` previously used $128 + 10 = 138$ (CBF) and $64 + 10 = 74$ (ATT). | **Requires Construction:** 79-dim concatenation for both CBF and ATT. |
| **ELM Architecture** | $H=512$ hidden units, $\tanh$ activation, random $W, b$, analytical ridge regression | `GPUChunkedELM` in `experiments/elm_models/elm_engine.py` | **Verified Identical** |
| **ELM Target** | Pure residual $r = y_{\text{true}} - \hat{y}_{\text{DNN}}$ | Multiple variants existed: direct parameter, OOF residual, damped residual. | **Verified Algorithm:** Learn $r = y - \hat{y}_{\text{DNN}}$, predict $\hat{y}_{\text{HDELM}} = \text{clip}(\hat{y}_{\text{DNN}} + \hat{r}, [y_{\min}, y_{\max}])$. |
| **Training SNR Grid** | 100 uniformly spaced rSNR levels from 10 to 20 ($N=1{,}000{,}000$) | `experiments/cache/snr_train.npy` uses 12 discrete bins ($[5, \dots, \infty]$). | **Data Variant:** Paper uses dense rSNR in $[10, 20]$; repo cache has 12 discrete levels. |

---

## 2. Detailed Forensic Inventory Across the 16 Audit Items

### Item 1: Original Buxton Kinetic Model Implementation
- **File Locations:**
  - `experiments/physics.py` (lines 35–49)
  - `asl-dnn-papermatched-v2-noise-density.ipynb` (Cell 4)
  - `dnn-elm-noise (7).ipynb` (Cell 4)
- **Formulation:**
  $$\Delta M(w) = 2 \cdot \alpha \cdot \beta \cdot T_{1t} \cdot \frac{1}{\lambda} \cdot f \cdot e^{-\delta / T_{1a}} \left[ e^{-\max(w - \delta, 0)/T_{1t}} - e^{-\max(\tau + w - \delta, 0)/T_{1t}} \right] \times \text{SCALE}$$
  where:
  - $w \in [1.525, 2.025, 2.525, 3.025]$ s
  - $\tau = 1.8$ s (labeling duration)
  - $T_{1t} = 1.2$ s (tissue longitudinal relaxation)
  - $T_{1a} = 1.66$ s (blood longitudinal relaxation)
  - $\alpha = 0.85$ (labeling efficiency)
  - $\beta = 0.75$ (arterial transit attenuation)
  - $\lambda = 0.9$ mL/g (blood-brain partition coefficient)
  - $\text{SCALE} = 100{,}000.0$
  - Perfusion conversion: $f = \frac{\text{CBF}}{6000 \cdot \lambda}$ s$^{-1}$
  - Transit time: $\delta = \text{ATT}$ s

### Item 2: ASL Signal Simulation Functions
- **File Location:** `experiments/physics.py`
- **Functions:**
  - `asl_signal(pld, tau_val, f_val, att_val)`: Scalar and array forward simulation.
  - `compute_physical_signals(plds, tau_val, cbf, att)`: Vectorized signal generation returning $(N, 4)$ array of clean signals.
  - Reference point: $\text{CBF}=50.0$, $\text{ATT}=1.6$, $\text{PLD}=2.0$, $\text{LD}=1.8 \implies \text{SIG\_REF\_SC} \approx 334.2039$.
  - Clean sanity check: $\text{CBF}=50, \text{ATT}=1.5 \implies [485.16653, 319.84150, 210.85252, 139.00255]$. Clean zero-noise input $(2\times S) = [970.33307, 639.68300, 421.70505, 278.00510]$. Verified exact match in code.

### Item 3: Rician Noise Generation Implementation
- **File Location:** `experiments/physics.py` (`add_rician_noise`, lines 81–110)
- **Formulation:**
  - Reference amplitude scalar: $\text{SIG\_REF\_SC} \approx 334.2039$.
  - Noise standard deviation: $\sigma = \frac{\text{SIG\_REF\_SC}}{\text{SNR}}$ (with $\sigma = 0$ if $\text{SNR} = \infty$).
  - Two independent complex noise channels with quadrature Gaussian components:
    $$e_1, e_2, e_3, e_4 \sim \mathcal{N}(0, \sigma^2)$$
    $$M_c = \sqrt{(S + e_1)^2 + e_2^2}$$
    $$M_l = \sqrt{(S + e_3)^2 + e_4^2}$$
    $$X = (M_c + M_l)$$
  - This models magnitude reconstruction of control and label acquisitions ($M_c + M_l$). Clean noiseless limit is $2S$.

### Item 4: CBF and ATT Sampling Distributions
- **Distributions:**
  - $\text{CBF} \sim \mathcal{U}(0.0, 100.0)$ mL/(100g $\cdot$ min)
  - $\text{ATT} \sim \mathcal{U}(0.5, 3.0)$ s
- **Implementation:** `rng.uniform(CBF_MIN, CBF_MAX, N)` and `rng.uniform(ATT_MIN, ATT_MAX, N)`.

### Item 5 & 6: Training and Validation Dataset Generation Scripts & Cache
- **File Location:** `experiments/dataset.py`, `experiments/cache/`
- **Existing Disk Cache:**
  - `experiments/cache/X_train.npy` ($1{,}000{,}000 \times 4$, float32, seed 42)
  - `experiments/cache/Y_train.npy` ($1{,}000{,}000 \times 2$, float32, seed 42)
  - `experiments/cache/snr_train.npy` ($1{,}000{,}000$, float64, 12 discrete bins: $[5, 8, 10, 12, 15, 20, 30, 50, 100, 200, 500, \infty]$)
  - `experiments/cache/X_val.npy` ($100{,}000 \times 4$, float32, seed 99)
  - `experiments/cache/Y_val.npy` ($100{,}000 \times 2$, float32, seed 99)
  - `experiments/cache/snr_val.npy` ($100{,}000$, float64)
- **Paper Specification Difference:**
  - Paper: $10{,}000$ unique pairs $\times 100$ rSNR levels uniformly spaced in $[10, 20]$ ($1{,}000{,}000$ samples, seed 42).
  - Validation: $1{,}000$ unique pairs $\times 100$ rSNR levels in $[10, 20]$ ($100{,}000$ samples, seed 99).
  - Notebook `asl-dnn-papermatched-v2-noise-density.ipynb` Cell 6 implements dense sweeping, but cached `.npy` files were generated with the 12-bin discrete distribution.

### Item 7: Original DNN-G Baseline Architecture
- **Repository Implementation:** `experiments/train_baseline_A.py` (`StandardizedNet`)
  - ATT: `StandardizedNet(4, n_hidden_layers=4, n_neurons=64, ...)`
  - CBF: `StandardizedNet(4, n_hidden_layers=5, n_neurons=128, ...)`
- **Research Paper Architecture:**
  - Both CBF and ATT must have 4 hidden layers $\times$ 64 neurons:
    $$\text{Linear}(4, 64) \to \text{ELU} \to \text{Linear}(64, 64) \to \text{ELU} \to \text{Linear}(64, 64) \to \text{ELU} \to \text{Linear}(64, 64) \to \text{ELU} \to \text{Linear}(64, 1)$$
  - Output transformation:
    $$\hat{y}_{\text{CBF}} = 0 + (100 - 0) \cdot \sigma(z)$$
    $$\hat{y}_{\text{ATT}} = 0.5 + (3.0 - 0.5) \cdot \sigma(z)$$
  - Training: Adam ($\text{lr}=10^{-3}$), normalized MSE loss, batch size 512, max epochs 200, gradient clipping 1.0, patience 10.

### Item 8: Residual H-DELM Implementation
- **Repository Implementations:**
  - `experiments/elm_models/run_elm_comparisons.py` (lines 255–316): Fits ELM on $[X_{\text{norm}}, \text{CBF}_{\text{DNN}}, \text{ATT}_{\text{DNN}}]$ to predict residuals with scalar shrinkage $\gamma$.
  - `experiments/elm_features_02/run_stage3_residual_elm.py`: Out-Of-Fold (OOF) 5-fold cross-validated residual ELM.
  - `experiments/elm_models/elm_engine.py`: `GPUChunkedELM` solver.
- **Paper Specification:**
  - Freeze the 4x64 DNN backbones.
  - Extract the 64-dimensional penultimate activations.
  - Concatenate with the 15 signal-derived physics features $\implies 79$ dimensions.
  - Standardize the 79-dimensional features using training statistics.
  - Fit ELM on residual targets $r = y_{\text{true}} - \hat{y}_{\text{DNN}}$.
  - Solve via ridge regression: $\beta = (H^T H + \lambda I)^{-1} H^T r$.
  - Final inference: $\hat{y}_{\text{HDELM}} = \text{clip}(\hat{y}_{\text{DNN}} + \hat{r}, [y_{\min}, y_{\max}])$.

### Item 9: The 15 Physics-Feature Extraction Implementation
- **Repository Implementation:** `experiments/elm_features_02/feature_engineering.py`
  - Defines `extract_physics_features()` with 10 features, and `extract_normalized_diff_features()` with 7 features.
- **Paper Requirement:** Exactly 15 physical features from the 4 PLD signals $[x_1, x_2, x_3, x_4]$ and delays $[w_1, w_2, w_3, w_4]$:
  1. 4 raw PLD signals: $x_1, x_2, x_3, x_4$
  2. 3 inter-PLD slopes: $\frac{x_2 - x_1}{w_2 - w_1}, \frac{x_3 - x_2}{w_3 - w_2}, \frac{x_4 - x_3}{w_4 - w_3}$ (where $\Delta w = 0.5$ s)
  3. 3 consecutive signal ratios: $\frac{x_2}{\max(x_1, \epsilon)}, \frac{x_3}{\max(x_2, \epsilon)}, \frac{x_4}{\max(x_3, \epsilon)}$ ($\epsilon = 10^{-3}$)
  4. Signal mean: $\mu_x = \frac{1}{4} \sum_{i=1}^4 x_i$
  5. Signal variance: $\sigma_x^2 = \frac{1}{4} \sum_{i=1}^4 (x_i - \mu_x)^2$
  6. Signal minimum: $\min(x_1, x_2, x_3, x_4)$
  7. Signal maximum: $\max(x_1, x_2, x_3, x_4)$
  8. Trapezoidal Area Under Curve (AUC): $\frac{1}{2} \sum_{i=1}^3 (x_i + x_{i+1})(w_{i+1} - w_i) = 0.25 (x_1 + 2x_2 + 2x_3 + x_4)$
  - Total: $4 + 3 + 3 + 1 + 1 + 1 + 1 + 1 = 15$ features.

### Item 10: ELM Random Projection & Ridge Regression Implementation
- **File Location:** `experiments/elm_models/elm_engine.py` (`GPUChunkedELM`)
- **Mathematical Specification:**
  - Hidden projection: $H = \tanh(Z W + b)$, where $Z \in \mathbb{R}^{N \times 79}$.
  - Weight initialization: $W_{ij}, b_j \sim \mathcal{U}(-1/\sqrt{d}, 1/\sqrt{d})$ with $d = 79$.
  - Hidden neurons: $L = 512$.
  - Output weight solve: $\beta = (H^T H + \lambda I)^{-1} H^T r$.
  - Numerical implementation: Cholesky or `torch.linalg.solve` in float64 precision.

### Item 11: Saved Model Checkpoints
- **Existing Checkpoints in Repository:**
  - `experiments/baseline_A/ATTnet_G_baselineA.pth` (4 hidden layers $\times$ 64 neurons, 55,570 bytes).
  - `experiments/baseline_A/CBFnet_G_baselineA.pth` (5 hidden layers $\times$ 128 neurons, 272,034 bytes).
- **Audit Finding:**
  - `ATTnet_G_baselineA.pth` has architecture `[4, 64, 64, 64, 64, 1]`, matching the paper's 4x64 backbone.
  - However, no 4x64 checkpoint for CBF exists anywhere in the repository.
  - Furthermore, `ATTnet_G_baselineA.pth` uses linear output on standardized target rather than the sigmoid output head.
  - **Decision:** To fulfill the critical requirement ("The model in the uploaded research paper has four hidden layers with 64 neurons each for BOTH CBF and ATT"), we must train the paper-matched 4x64 networks with the sigmoid-bounded head for both CBF and ATT using seed 42 on the training dataset.

### Item 12: Input Normalization Statistics
- **File Locations:** `experiments/baseline_A/normalization_stats.npz`, `experiments/cache/normalization_stats.npz`
- **Values (fitted on $1{,}000{,}000$ training samples, seed 42):**
  - $X_{\text{mean}} = [656.0311, 560.5862, 435.96695, 308.37015]$
  - $X_{\text{std}} = [480.29407, 344.71332, 253.01886, 179.04381]$

### Item 13: Target Normalization and Physiological Bounds
- **Bounds:**
  - CBF: $[0.0, 100.0]$ mL/100g/min
  - ATT: $[0.5, 3.0]$ s
- **Statistics in Cache:**
  - CBF: $\mu = 50.0010, \sigma = 28.8671$
  - ATT: $\mu = 1.7516, \sigma = 0.7221$
- **Sigmoid Normalized Target:**
  $$y_{\text{norm}} = \frac{y - y_{\min}}{y_{\max} - y_{\min}} \in [0, 1]$$
  Loss: $\text{MSE}(\sigma(z), y_{\text{norm}})$. Physical output: $y_{\min} + (y_{\max} - y_{\min}) \cdot \sigma(z)$.

### Item 14: ELM Feature Normalization Statistics
- Standardize the 79-dimensional features ($F \in \mathbb{R}^{N \times 79}$):
  $$\mu_F = \frac{1}{N} \sum_{i=1}^N F_i, \quad \sigma_F = \sqrt{\frac{1}{N} \sum_{i=1}^N (F_i - \mu_F)^2}$$
  $$F_{\text{norm}} = \frac{F - \mu_F}{\sigma_F + 10^{-6}}$$
- Computed strictly on the training set.

### Item 15: ELM Ridge Regularization Parameters
- Historical sweep in `experiments/elm_models/run_elm_comparisons.py` tested $\lambda \in [10^{-4}, 1.0]$.
- Standard parameter: $\lambda = 10^{-2}$ ($0.01$). Validation sweep over $\{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0\}$ confirms $\lambda = 10^{-2}$ is optimal.

### Item 16: Existing Experiment Configurations and Random Seeds
- **Simulation Seeds:**
  - Train: seed 42
  - Validation: seed 99
  - Locked test in cache: seed 7
- **Model Training Seeds:**
  - PyTorch manual seed: 42
  - ELM projection seed: 42

---

## 3. Missing Components & Action Plan

1. **4x64 Bounded Sigmoid DNN Backbone:**
   Construct and train the exact 4 hidden layer $\times$ 64 neuron DNN-G model for both CBF and ATT using Adam, lr=1e-3, batch size 512, max 200 epochs, early stopping patience 10, saving best validation checkpoints.
2. **Exact 15 Physics-Feature Extractor:**
   Implement `extract_15_physics_features(X_raw)` extracting the 8 exact feature groups.
3. **Residual H-DELM Model:**
   Implement frozen 4x64 backbone + 64-dim penultimate feature extraction + 15 physics features $\to$ 79-dim standardized input $\to$ 512-neuron $\tanh$ ELM predicting residual $r = y - \hat{y}_{\text{DNN}} \to$ analytical ridge regression $\beta \to$ physical clipping.
4. **Independent 30,000-Observation Test Dataset:**
   Generate 10,000 observations at SNR 10, 10,000 at SNR 15, and 10,000 at SNR 20 with completely independent ground truth parameters ($\mathcal{U}[0, 100]$ for CBF, $\mathcal{U}[0.5, 3.0]$ for ATT), reference Rician noise calibration ($\text{SIG\_REF\_SC} \approx 334.2039$), using a new held-out seed (seed 2026).
5. **Evaluation, Tables & Figures:**
   Compute MSE, RMSE, MAE, and Lin's Concordance Correlation Coefficient (CCC) for both models at SNR 10, 15, and 20; paired bootstrap significance testing; generate publication-quality figures (PNG 300 DPI and PDF).

