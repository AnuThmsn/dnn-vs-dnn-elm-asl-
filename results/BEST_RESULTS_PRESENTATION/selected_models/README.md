# Selected Models Manifest & Architecture Guide

This directory defines the exact model specifications and frozen checkpoint references.

## 1. Frozen Baseline DNN-G
- Checkpoints: `experiments/baseline_A/CBFnet_G_baselineA.pth` and `ATTnet_G_baselineA.pth`
- Normalization: `experiments/baseline_A/normalization_stats.npz`
- CBF: 5 hidden layers of 128 units, ELU activations.
- ATT: 4 hidden layers of 64 units, ELU activations.

## 2. Primary Defensible DNN+ELM: Ensemble Feature ELM
- Taps the penultimate activations ($h_5$ for CBF, $h_4$ for ATT).
- Hidden layer: 512 units with $\tanh$ activation.
- Regularization: $\alpha = 10^{-2}$.
- Ensemble: 5 independent random weight initializations (seeds 42, 43, 44, 45, 46).
