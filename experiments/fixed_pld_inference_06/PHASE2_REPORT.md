# Phase 2 Report: Physics-Based Initial Estimates and Learned Refinement

## 1. Executive Summary
Phase 2 designed and evaluated physics-informed input representations, a linear GKM initial estimator, and learned refinement architectures. All transformations were strictly fitted on the training split without data leakage.

## 2. Validation Results Comparison

| Model | CBF_RMSE | CBF_MAE | CBF_CCC | ATT_RMSE | ATT_MAE | ATT_CCC |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline A DNN-G | 3.7411 | 2.5453 | 0.9915 | 0.3042 | 0.1934 | 0.9028 |
| Ensemble Feature ELM | 3.7105 | 2.4515 | 0.9916 | 0.3046 | 0.1913 | 0.9028 |
| Physics Initial Estimator Alone | 5.1850 | 3.6501 | 0.9835 | 0.4374 | 0.3416 | 0.7723 |
| Full Physics Features ELM | 3.7683 | 2.5360 | 0.9914 | 0.3115 | 0.2019 | 0.8980 |
| Hybrid (Penultimate + Physics ELM) | 3.7327 | 2.4781 | 0.9915 | 0.3048 | 0.1929 | 0.9024 |


## 3. Feature Ablation Study

| Feature_Set | Dim | CBF_RMSE | CBF_MAE | CBF_CCC | CBF_R2 | ATT_RMSE | ATT_MAE | ATT_CCC | ATT_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw Signals (4 dims) | 4.0000 | 3.8180 | 2.6298 | 0.9911 | 0.9825 | 0.3197 | 0.2119 | 0.8915 | 0.8043 |
| Raw + Safe Log (8 dims) | 8.0000 | 3.7941 | 2.5982 | 0.9913 | 0.9827 | 0.3127 | 0.2060 | 0.8968 | 0.8128 |
| Raw + Ratios + Branch (10 dims) | 10.0000 | 3.7709 | 2.5340 | 0.9914 | 0.9829 | 0.3108 | 0.2011 | 0.8984 | 0.8151 |
| Full Physics Features (16 dims) | 16.0000 | 3.7683 | 2.5360 | 0.9914 | 0.9829 | 0.3115 | 0.2019 | 0.8980 | 0.8142 |


## 4. Key Findings
- **Physics-Based Initial Estimator Alone:** A purely linear inversion achieves CBF RMSE of `5.1850` and ATT RMSE of `0.4374`. While competent for CBF due to quasi-linearity, it fails on ATT because ATT enters exponentially in kinetic decay.
- **Physics Features vs Penultimate Representations:** Directly training an ELM on raw signals plus log-ratios and branch indicators achieves CBF RMSE of `3.7683`. The deep backbone of Baseline A DNN-G produces internal representations that are superior, achieving `3.7411`.
- **Hybrid Concatenation:** Concatenating penultimate features with physics indicators yields CBF RMSE of `3.7327` (essentially identical to penultimate ELM `3.7105`). This confirms that the 5-layer DNN backbone has already implicitly synthesized these nonlinear signal ratios.
