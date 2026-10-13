# Stage 1 — DNN Architecture and Hidden Layer Audit

**Experiment:** Experiment 4 — Multi-Layer Feature ELM for ASL-MRI Parameter Estimation  
**Repository:** `https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-`  
**Date:** 2026-10-13  
**Status:** Complete & Non-Invasively Verified  

---

## 1. Executive Summary & Audit Objectives

Experiment 4 investigates whether combining intermediate hidden layers with penultimate layer representations, or applying alternative nonlinear readout mechanisms (ELU, ReLU, Random Fourier Feature Kernel-ELM), can improve ASL-MRI parameter estimation (CBF and ATT) over the frozen Baseline A DNN and the historical single-layer Feature ELM.

All experiments are conducted under the strictly frozen, standard 4-PLD acquisition protocol:
- **PLDs:** $[1.525, 2.025, 2.525, 3.025]$ seconds
- **Label Duration:** $\tau = 1.8$ seconds
- **Physical Reference Constants:** $T_{1,\text{blood}} = 1.66\text{ s}$, $T_{1,\text{tissue}} = 1.20\text{ s}$, $\alpha = 0.85$, $\beta = 0.75$, $\lambda = 0.90\text{ mL/g}$
- **Noise Model:** Dual-channel Rician noise across 12 standard SNR levels ($5.0$ to $\infty\text{ dB}$)
- **Target Physiological Bounds:** $\text{CBF} \in [0, 100]\text{ mL}/(100\text{g}\cdot\text{min})$, $\text{ATT} \in [0.5, 3.0]\text{ s}$

---

## 2. Frozen Base DNN Architectural Specifications

The base estimators are frozen from `experiments/baseline_A/`:
- CBF Checkpoint: `experiments/baseline_A/checkpoints/CBFnet_G_baselineA.pth` (or root `experiments/baseline_A/CBFnet_G_baselineA.pth`)
- ATT Checkpoint: `experiments/baseline_A/checkpoints/ATTnet_G_baselineA.pth` (or root `experiments/baseline_A/ATTnet_G_baselineA.pth`)
- Normalization Stats: `experiments/baseline_A/normalization_stats.npz`

### 2.1 CBF Network (`CBFnet_G_baselineA`)
- **Input Dimension:** 4 ($X \in \mathbb{R}^4$, standardized by global training statistics $X_{\text{mean}}, X_{\text{std}}$)
- **Architecture Backbone:** 5 hidden layers, 128 units each, with Exponential Linear Unit (ELU, $\alpha=1.0$) activations:
  - `Layer 0`: `Linear(4, 128)`
  - `Layer 1`: `ELU(alpha=1.0)` $\rightarrow$ **$h_1$ activation** ($\mathbb{R}^{128}$, early representation)
  - `Layer 2`: `Linear(128, 128)`
  - `Layer 3`: `ELU(alpha=1.0)` $\rightarrow$ **$h_2$ activation** ($\mathbb{R}^{128}$, intermediate representation)
  - `Layer 4`: `Linear(128, 128)`
  - `Layer 5`: `ELU(alpha=1.0)` $\rightarrow$ **$h_3$ activation** ($\mathbb{R}^{128}$, intermediate representation)
  - `Layer 6`: `Linear(128, 128)`
  - `Layer 7`: `ELU(alpha=1.0)` $\rightarrow$ **$h_4$ activation** ($\mathbb{R}^{128}$, pre-penultimate representation)
  - `Layer 8`: `Linear(128, 128)`
  - `Layer 9`: `ELU(alpha=1.0)` $\rightarrow$ **$h_5$ activation** ($\mathbb{R}^{128}$, penultimate representation — historical baseline)
  - `Layer 10`: `Linear(128, 1)` $\rightarrow$ Standardized scalar output prediction
- **Target Un-scaling:** $\hat{y}_{\text{phys}} = \text{clamp}(\hat{y}_{\text{std}} \cdot \sigma_y + \mu_y, 0, 100)$ where $\mu_y = 50.0010$, $\sigma_y = 28.8671$.

### 2.2 ATT Network (`ATTnet_G_baselineA`)
- **Input Dimension:** 4 ($X \in \mathbb{R}^4$, standardized by global training statistics $X_{\text{mean}}, X_{\text{std}}$)
- **Architecture Backbone:** 4 hidden layers, 64 units each, with Exponential Linear Unit (ELU, $\alpha=1.0$) activations:
  - `Layer 0`: `Linear(4, 64)`
  - `Layer 1`: `ELU(alpha=1.0)` $\rightarrow$ **$h_1$ activation** ($\mathbb{R}^{64}$, early representation)
  - `Layer 2`: `Linear(64, 64)`
  - `Layer 3`: `ELU(alpha=1.0)` $\rightarrow$ **$h_2$ activation** ($\mathbb{R}^{64}$, intermediate representation)
  - `Layer 4`: `Linear(64, 64)`
  - `Layer 5`: `ELU(alpha=1.0)` $\rightarrow$ **$h_3$ activation** ($\mathbb{R}^{64}$, pre-penultimate representation)
  - `Layer 6`: `Linear(64, 64)`
  - `Layer 7`: `ELU(alpha=1.0)` $\rightarrow$ **$h_4$ activation** ($\mathbb{R}^{64}$, penultimate representation — historical baseline)
  - `Layer 8`: `Linear(64, 1)` $\rightarrow$ Standardized scalar output prediction
- **Target Un-scaling:** $\hat{y}_{\text{phys}} = \text{clamp}(\hat{y}_{\text{std}} \cdot \sigma_y + \mu_y, 0.5, 3.0)$ where $\mu_y = 1.7516$, $\sigma_y = 0.7221$.

---

## 3. Non-Invasive Feature Extraction Verification

To guarantee zero alteration of the base neural network predictions, intermediate representations are tapped using clean, forward-hook-free sequential accumulation during the forward pass:
- An audit script (`stage1_architecture_audit.py`) was executed on validation samples ($N=1,000$).
- Output predictions from the multi-layer extractor were compared directly against the standalone `StandardizedNet.predict_physical()` method.
- Penultimate extracted features were compared directly against `StandardizedNet.penultimate_features()`.

### Numerical Verification Results:
| Metric | CBF Net | ATT Net | Verification Status |
| :--- | :---: | :---: | :---: |
| Max Pred Absolute Diff | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASSED** (Exact bit-level identity) |
| Max Penultimate Feat Diff | $0.00 \times 10^0$ | $0.00 \times 10^0$ | **PASSED** (Exact bit-level identity) |

---

## 4. Multi-Layer Representations Defined for Stage 2

Five representations are formulated for evaluation:

1. **Representation A (Control / Historical Baseline):**
   - CBF: Penultimate layer $h_5 \in \mathbb{R}^{128}$
   - ATT: Penultimate layer $h_4 \in \mathbb{R}^{64}$
2. **Representation B (Adjacent Hidden Layers):**
   - CBF: Concatenation $[h_4, h_5] \in \mathbb{R}^{256}$
   - ATT: Concatenation $[h_3, h_4] \in \mathbb{R}^{128}$
3. **Representation C (Early + Penultimate Layers):**
   - CBF: Concatenation $[h_1, h_5] \in \mathbb{R}^{256}$
   - ATT: Concatenation $[h_1, h_4] \in \mathbb{R}^{128}$
4. **Representation D (Normalized Multi-Layer Representation):**
   - CBF: Concatenation of $[h_1, h_3, h_5] \in \mathbb{R}^{384}$ with layer-wise $L_2$ unit normalization: $\bar{h}_i = h_i / (\|h_i\|_2 + \epsilon)$
   - ATT: Concatenation of $[h_1, h_2, h_4] \in \mathbb{R}^{192}$ with layer-wise $L_2$ unit normalization: $\bar{h}_i = h_i / (\|h_i\|_2 + \epsilon)$
5. **Representation E (Compact PCA Projection of Full Hidden Latent Space):**
   - CBF: Concatenation $[h_1, h_2, h_3, h_4, h_5] \in \mathbb{R}^{640}$ projected to $128$ dimensions using PCA fitted strictly on training data
   - ATT: Concatenation $[h_1, h_2, h_3, h_4] \in \mathbb{R}^{256}$ projected to $64$ dimensions using PCA fitted strictly on training data

---

## 5. Non-Linear Readouts Defined for Stage 3

Using the best-performing representations from Stage 2:
1. **Ridge ELM (Tanh activation):** $H = \tanh(X W_{\text{rand}} + b_{\text{rand}})$, with closed-form regularized least squares $\beta = (H^T H + \lambda I)^{-1} H^T Y$.
2. **ELM with Alternative Activations:**
   - **ELU activation:** $H = \text{ELU}(X W_{\text{rand}} + b_{\text{rand}})$ (matching DNN backbone activation).
   - **ReLU activation:** $H = \max(0, X W_{\text{rand}} + b_{\text{rand}})$.
3. **Kernel-ELM / Random Fourier Features (RFF Kernel Ridge Regression):**
   - Explicit randomized mapping $z(x) = \sqrt{\frac{2}{D}} \cos(X W_{\text{rff}} + b_{\text{rff}})$ where $W_{\text{rff}} \sim \mathcal{N}(0, \sigma_{\text{rbf}}^{-2} I)$ and $b_{\text{rff}} \sim \mathcal{U}[0, 2\pi]$, approximating the Gaussian RBF kernel $k(x, x') = \exp(-\|x - x'\|^2 / (2 \sigma^2))$ without $\mathcal{O}(N^2)$ memory explosion.
4. **Compact Regularized Benchmark Readout:**
   - Linear ridge regression directly on the multi-layer representations without random hidden projection, isolating the pure linear readout benefit.

---

## 6. Execution Plan & Leakage-Free Protocol

- **Data Partitions:**
  - `X_train.npy` / `Y_train.npy`: $N=2,000,000$ (training split)
  - `X_val.npy` / `Y_val.npy`: $N=100,000$ (validation split, used for all model selection and hyperparameter tuning)
  - `X_test.npy` / `Y_test.npy`: $N=100,000$ (locked test set, accessed strictly in Stage 5)
- **Computational Policy:**
  - Chunked GPU matrix accumulation ($N_{\text{chunk}} = 25,000$) to strictly respect $< 80\text{ MB}$ memory footprint and prevent CUDA out-of-memory errors on NVIDIA GTX 1650 (4GB VRAM).
- **Git Commit Tracking:**
  - Stage 1: Commit architecture audit and `config.json`.
  - Stage 2: Commit representation comparison script and `stage2_representation_comparison.csv`.
  - Stage 3: Commit nonlinear readout script and `stage3_nonlinear_comparison.csv`.
  - Stage 4: Commit validation report, diagnostics, and plots.
  - Stage 5: Commit locked-test evaluation, final test metrics, and `FINAL_REPORT.md`.
