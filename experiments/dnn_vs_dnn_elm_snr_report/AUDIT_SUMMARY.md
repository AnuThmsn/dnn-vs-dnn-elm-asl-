# Phase 1 Audit Summary: DNN and DNN-ELM Benchmark Artifacts

## 1. Experimental Protocol & Frozen Constants
- **Acquisition Protocol:** Fixed original 4-PLD pCASL (`[1.525, 2.025, 2.525, 3.025]` s, $\tau = 1.8$ s).
- **Physical Constants:** $T_{1,\text{blood}} = 1.66$ s, $T_{1,\text{tissue}} = 1.20$ s, $\alpha = 0.85$, $\beta = 0.75$, $\lambda = 0.90$ mL/g, Scale = $100{,}000$.
- **Noise Model:** Dual-channel quadrature Rician magnitude noise simulation (`add_rician_noise` in `physics.py`).

## 2. Models Verified
### A. Baseline DNN (DNN-G)
- **CBFnet:** 5 hidden layers x 128 neurons, ELU activations (67,841 parameters). Checkpoint: `CBFnet_G_baselineA.pth`.
- **ATTnet:** 4 hidden layers x 64 neurons, ELU activations (13,057 parameters). Checkpoint: `ATTnet_G_baselineA.pth`.
- **Normalization:** Input standardization via `(X - X_mean) / X_std`; target standardization via `(Y - y_mean) / y_std`.

### B. DNN-ELM Variant Definition
- **Architecture:** **Feature ELM** operating directly on the penultimate hidden layer representations of the frozen baseline DNN.
- **Readout:** Single layer with $H = 512$ hidden units, $\tanh$ activation function, and $L_2$ Ridge regression regularizer $\alpha = 10^{-2}$.
- **Variants Evaluated:** Both **Single-Seed Feature ELM (seed 42)** and the authoritative **5-Seed Ensemble Feature ELM (seeds 42..46)** are evaluated to report seed variability.

## 3. Evaluated Test Cohort at SNR 10, 15, and 20 dB
- **Total locked test set:** $N = 100{,}000$ synthetic test voxels.
- **SNR 10.0 dB:** $N = 8,358$ test samples.
- **SNR 15.0 dB:** $N = 8,279$ test samples.
- **SNR 20.0 dB:** $N = 8,385$ test samples.
- **Integrity:** Zero overlap between training/validation and the locked test set. Checkpoint hashes confirmed.
