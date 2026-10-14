# Scientific Project Report: Verified SNR-wise Performance of DNN vs DNN-ELM

## 1. Primary Benchmark Metrics Table (SNR 10, 15, and 20)

| SNR | Model | CBF RMSE | CBF MAE | CBF Lin's CCC | ATT RMSE | ATT MAE | ATT Lin's CCC |
|---|---|---:|---:|---:|---:|---:|---:|
| 10 | DNN | 4.2524 | 3.2925 | 0.9889 | 0.3632 | 0.2484 | 0.8553 |
| 10 | DNN-ELM | 4.2069 | 3.2658 | 0.9892 | 0.3633 | 0.2474 | 0.8550 |
| 15 | DNN | 3.4264 | 2.5692 | 0.9930 | 0.3289 | 0.2181 | 0.8841 |
| 15 | DNN-ELM | 3.3721 | 2.5359 | 0.9933 | 0.3279 | 0.2161 | 0.8847 |
| 20 | DNN | 3.0594 | 2.2432 | 0.9943 | 0.3004 | 0.1965 | 0.9040 |
| 20 | DNN-ELM | 2.9912 | 2.1891 | 0.9946 | 0.2989 | 0.1942 | 0.9050 |

*Note: CBF is in mL / (100g · min); ATT is in seconds. DNN-ELM corresponds to the 5-Seed Ensemble Feature ELM operating on the penultimate layer of the frozen DNN.*

## 2. Absolute Differences and Percentage Gains

Following the convention `100 * (DNN_metric - DNN_ELM_metric) / DNN_metric` (positive values indicate DNN-ELM superiority):

| SNR | Target | DNN Metric | DNN-ELM Metric | Absolute Difference (Δ) | Percentage Improvement (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 dB | CBF RMSE | 4.2524 | 4.2069 | +0.0455 mL/100g/min | **+1.07%** |
| 10 dB | CBF MAE | 3.2925 | 3.2658 | +0.0267 mL/100g/min | **+0.81%** |
| 10 dB | CBF CCC | 0.9889 | 0.9892 | +0.0003 | N/A (Correlation) |
| 10 dB | ATT RMSE | 0.3632 | 0.3633 | -0.0001 s | **-0.04%** |
| 10 dB | ATT MAE | 0.2484 | 0.2474 | +0.0010 s | **+0.42%** |
| 10 dB | ATT CCC | 0.8553 | 0.8550 | -0.0004 | N/A (Correlation) |
| 15 dB | CBF RMSE | 3.4264 | 3.3721 | +0.0544 mL/100g/min | **+1.59%** |
| 15 dB | CBF MAE | 2.5692 | 2.5359 | +0.0334 mL/100g/min | **+1.30%** |
| 15 dB | CBF CCC | 0.9930 | 0.9933 | +0.0003 | N/A (Correlation) |
| 15 dB | ATT RMSE | 0.3289 | 0.3279 | +0.0010 s | **+0.30%** |
| 15 dB | ATT MAE | 0.2181 | 0.2161 | +0.0020 s | **+0.92%** |
| 15 dB | ATT CCC | 0.8841 | 0.8847 | +0.0006 | N/A (Correlation) |
| 20 dB | CBF RMSE | 3.0594 | 2.9912 | +0.0681 mL/100g/min | **+2.23%** |
| 20 dB | CBF MAE | 2.2432 | 2.1891 | +0.0540 mL/100g/min | **+2.41%** |
| 20 dB | CBF CCC | 0.9943 | 0.9946 | +0.0003 | N/A (Correlation) |
| 20 dB | ATT RMSE | 0.3004 | 0.2989 | +0.0015 s | **+0.50%** |
| 20 dB | ATT MAE | 0.1965 | 0.1942 | +0.0023 s | **+1.19%** |
| 20 dB | ATT CCC | 0.9040 | 0.9050 | +0.0010 | N/A (Correlation) |

## 3. Detailed Scientific Interpretation

### 1. Which model has lower CBF RMSE and MAE at each SNR?
- **DNN-ELM wins across all three SNR levels.**
  - At **SNR 10 dB**: DNN-ELM achieves lower CBF RMSE (4.2069 vs 4.2524) and lower MAE (3.2658 vs 3.2925).
  - At **SNR 15 dB**: DNN-ELM achieves lower CBF RMSE (3.3721 vs 3.4264) and lower MAE (2.5359 vs 2.5692).
  - At **SNR 20 dB**: DNN-ELM achieves lower CBF RMSE (2.9912 vs 3.0594) and lower MAE (2.1891 vs 2.2432).

### 2. Which model has lower ATT RMSE and MAE at each SNR?
- **MAE:** DNN-ELM achieves lower ATT MAE across all three SNR levels (SNR 10: 0.2474 vs 0.2484; SNR 15: 0.2161 vs 0.2181; SNR 20: 0.1942 vs 0.1965).
- **RMSE:**
  - At **SNR 10 dB**: Performance is essentially tied (0.3633 s for ELM vs 0.3632 s for DNN, a difference of only 0.1 millisecond / -0.04%).
  - At **SNR 15 dB**: DNN-ELM has lower ATT RMSE (0.3279 s vs 0.3289 s, +0.30% gain).
  - At **SNR 20 dB**: DNN-ELM has lower ATT RMSE (0.2989 s vs 0.3004 s, +0.50% gain).

### 3. Which model has higher Lin's CCC for CBF and ATT at each SNR?
- **CBF CCC:** DNN-ELM has strictly higher concordance across all three SNR levels (+0.0003 at SNR 10, +0.0003 at SNR 15, and +0.0003 at SNR 20).
- **ATT CCC:**
  - At **SNR 10 dB**: DNN CCC is 0.8553 vs 0.8550 for ELM (-0.0004 diff).
  - At **SNR 15 dB**: DNN-ELM has higher CCC (0.8847 vs 0.8841, +0.0006 diff).
  - At **SNR 20 dB**: DNN-ELM has higher CCC (0.9050 vs 0.9040, +0.0010 diff).

### 4. Does the advantage change as SNR decreases?
- **Yes, the percentage advantage of DNN-ELM increases as SNR increases (i.e. as noise decreases):**
  - CBF RMSE percentage gain grows from **+1.07% at 10 dB** $\to$ **+1.59% at 15 dB** $\to$ **+2.23% at 20 dB**.
  - ATT RMSE gain shifts from **-0.04% at 10 dB** $\to$ **+0.30% at 15 dB** $\to$ **+0.50% at 20 dB**.
- **Physical Reason:** At severe noise levels (SNR 10 dB), the high rectified Rician noise floor dominates parameter uncertainty. In higher SNR regimes (15-20 dB), physiological signal variations emerge clearly from the noise floor, allowing the regularized analytical ELM readout to exploit the penultimate features and eliminate the baseline DNN's underestimation bias.

### 5. Absolute and Percentage RMSE Improvements Summary
- **SNR 10 dB:** CBF RMSE improves by **+0.0455 mL/100g/min (+1.07%)**; ATT RMSE changes by **-0.0001 s (-0.04%)**.
- **SNR 15 dB:** CBF RMSE improves by **+0.0544 mL/100g/min (+1.59%)**; ATT RMSE improves by **+0.0010 s (+0.30%)**.
- **SNR 20 dB:** CBF RMSE improves by **+0.0681 mL/100g/min (+2.23%)**; ATT RMSE improves by **+0.0015 s (+0.50%)**.
