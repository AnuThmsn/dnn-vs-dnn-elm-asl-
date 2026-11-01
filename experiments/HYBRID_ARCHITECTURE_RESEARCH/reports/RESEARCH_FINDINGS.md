# Research findings: core hybrid architecture comparisons

## Scope and protocol qualifications

- DNN-G was reproduced from the frozen checkpoint; the maximum test-prediction difference from the saved 30K archive was exactly zero.
- Calibration-only fitting used 80,000 rows, with 20,000 held out for validation. No test observations were used to fit, tune, early-stop, or select a model.
- The 30K benchmark is the exact pre-existing independent test from the H-DELM submission and has already appeared in its report; it is not a new unseen test for this research phase.
- The attached BEST_RESULTS_PRESENTATION CSV has a different SNR grid and counts and was not pooled into this analysis.
- This is a controlled core comparison, not completion of every requested research phase. Results are descriptive because this benchmark has previously been used for the submitted H-DELM evaluation.

## Main results and interpretation

| Model / summary | Overall CBF RMSE | Overall ATT RMSE | Interpretation |
|---|---:|---:|---|
| DNN-G | 3.488532 | 0.321938 | Frozen baseline |
| Saved H-DELM V2 | 3.449308 | 0.322202 | CBF improves by 1.124%; ATT RMSE is 0.082% worse |
| Residual ELM, mean ± SD across 3 projection seeds | 3.451114 ± 0.000290 | 0.322567 ± 0.000238 | CBF improves by about 1.073%; ATT is worse by about 0.196% |
| Residual MLP, mean ± SD across 3 initialization seeds | 3.450111 ± 0.002291 | 0.321615 ± 0.000054 | Small mean gains for both overall RMSEs; CBF degrades at SNR 20 for all three seeds |

The saved H-DELM lowers CBF RMSE at SNR 10, 15, and 20 (1.789%, 0.786%,
0.278%), but ATT RMSE is slightly worse at all three. After Holm correction,
the H-DELM CBF comparisons are significant at SNR 10 (Wilcoxon adjusted
p=4.40e-35; paired permutation adjusted p=0.0210) and SNR 15 (6.97e-05;
0.0210), but not SNR 20 or ATT. Bootstrap 95% intervals for CBF RMSE
improvement are [0.0608, 0.0857] at SNR 10, [0.0146, 0.0369] at SNR 15, and
[-0.0028, 0.0189] at SNR 20.

The residual ELM beats the direct-output ELM for both targets and at all three
SNRs under the fixed architecture and ridge setting. Feature ablations do not
show that combining learned and physics features gives a consistent,
substantial advantage over either feature group alone: the combined model has
slightly lower overall CBF RMSE than representation-only ELM, but slightly
higher ATT RMSE. The calibration-validation alpha selection chooses 1.0 for
both parameters; this does not establish that full-strength correction is
universally optimal.

Across the three residual-MLP initializations, average overall errors are
close and the per-SNR CBF behavior is not uniform: mean CBF RMSE improves at
SNR 10 and 15, but is slightly worse at SNR 20. The small overall mean
improvement is therefore promising, not proof of broad superiority. The three
seeds vary only MLP initialization or ELM random projection; they do not
represent independent DNN training runs.

**Current answer to “which architecture is best?”:** no universal winner is
established. The saved H-DELM is the strongest tested reference for consistent
CBF improvement across SNRs; the three-seed residual MLP has a small mean
overall ATT benefit and similar CBF performance, but not a consistent CBF
benefit at SNR 20. These are exploratory comparisons on the previously used
fixed benchmark, not a final model-selection claim.

## Per-SNR results

| Model | SNR | Parameter | RMSE | MAE | Bias | CCC |
|---|---:|---|---:|---:|---:|---:|
| DNN-G | 10 | CBF | 4.109171 | 3.223274 | +0.897259 | 0.989487 |
| DNN-G | 10 | ATT | 0.357899 | 0.232657 | +0.018324 | 0.861893 |
| DNN-G | 15 | CBF | 3.282142 | 2.462446 | +0.200837 | 0.993552 |
| DNN-G | 15 | ATT | 0.308743 | 0.202120 | +0.006078 | 0.899215 |
| DNN-G | 20 | CBF | 2.975200 | 2.166215 | +0.040636 | 0.994690 |
| DNN-G | 20 | ATT | 0.295834 | 0.192704 | -0.006191 | 0.908470 |
| H-DELM-V2-saved | 10 | CBF | 4.035671 | 3.160971 | +0.560263 | 0.989882 |
| H-DELM-V2-saved | 10 | ATT | 0.358289 | 0.233326 | +0.014865 | 0.860951 |
| H-DELM-V2-saved | 15 | CBF | 3.256349 | 2.437992 | -0.087133 | 0.993668 |
| H-DELM-V2-saved | 15 | ATT | 0.309035 | 0.202825 | +0.001637 | 0.898684 |
| H-DELM-V2-saved | 20 | CBF | 2.966939 | 2.165984 | -0.227402 | 0.994731 |
| H-DELM-V2-saved | 20 | ATT | 0.295922 | 0.192874 | -0.010317 | 0.908099 |
| combined-features-residual-seed-42 | 10 | CBF | 4.038287 | 3.163461 | +0.563602 | 0.989865 |
| combined-features-residual-seed-42 | 10 | ATT | 0.358626 | 0.233708 | +0.014476 | 0.860796 |
| combined-features-residual-seed-42 | 15 | CBF | 3.257817 | 2.439154 | -0.088212 | 0.993661 |
| combined-features-residual-seed-42 | 15 | ATT | 0.308984 | 0.202998 | +0.001125 | 0.898779 |
| combined-features-residual-seed-42 | 20 | CBF | 2.967133 | 2.165562 | -0.226405 | 0.994729 |
| combined-features-residual-seed-42 | 20 | ATT | 0.296007 | 0.193071 | -0.011025 | 0.908098 |
| combined-features-direct | 10 | CBF | 4.160835 | 3.274629 | +0.492740 | 0.989235 |
| combined-features-direct | 10 | ATT | 0.360170 | 0.236349 | +0.013477 | 0.859816 |
| combined-features-direct | 15 | CBF | 3.316580 | 2.503428 | -0.091284 | 0.993430 |
| combined-features-direct | 15 | ATT | 0.309641 | 0.204660 | +0.000830 | 0.898395 |
| combined-features-direct | 20 | CBF | 3.012154 | 2.218247 | -0.203128 | 0.994569 |
| combined-features-direct | 20 | ATT | 0.296512 | 0.194395 | -0.010806 | 0.907791 |
| DNN-residual-MLP-128-64-seed-11 | 10 | CBF | 4.037759 | 3.157066 | +0.625339 | 0.989918 |
| DNN-residual-MLP-128-64-seed-11 | 10 | ATT | 0.357582 | 0.232073 | +0.020777 | 0.861557 |
| DNN-residual-MLP-128-64-seed-11 | 15 | CBF | 3.259719 | 2.440596 | -0.049989 | 0.993676 |
| DNN-residual-MLP-128-64-seed-11 | 15 | ATT | 0.308837 | 0.201733 | +0.009136 | 0.898751 |
| DNN-residual-MLP-128-64-seed-11 | 20 | CBF | 2.968277 | 2.170265 | -0.200308 | 0.994742 |
| DNN-residual-MLP-128-64-seed-11 | 20 | ATT | 0.295155 | 0.191571 | -0.002878 | 0.908503 |
| DNN-residual-MLP-128-64-seed-42 | 10 | CBF | 4.015174 | 3.147372 | +0.570526 | 0.989993 |
| DNN-residual-MLP-128-64-seed-42 | 10 | ATT | 0.357437 | 0.232008 | +0.015936 | 0.861416 |
| DNN-residual-MLP-128-64-seed-42 | 15 | CBF | 3.262510 | 2.445913 | -0.110350 | 0.993646 |
| DNN-residual-MLP-128-64-seed-42 | 15 | ATT | 0.308769 | 0.201966 | +0.004164 | 0.898563 |
| DNN-residual-MLP-128-64-seed-42 | 20 | CBF | 2.981536 | 2.183487 | -0.260245 | 0.994680 |
| DNN-residual-MLP-128-64-seed-42 | 20 | ATT | 0.295422 | 0.191903 | -0.007899 | 0.908149 |
| DNN-residual-MLP-128-64-seed-31415 | 10 | CBF | 4.019339 | 3.141487 | +0.650576 | 0.990010 |
| DNN-residual-MLP-128-64-seed-31415 | 10 | ATT | 0.357234 | 0.231989 | +0.015651 | 0.861663 |
| DNN-residual-MLP-128-64-seed-31415 | 15 | CBF | 3.263924 | 2.442991 | -0.055524 | 0.993667 |
| DNN-residual-MLP-128-64-seed-31415 | 15 | ATT | 0.308701 | 0.201928 | +0.004064 | 0.898679 |
| DNN-residual-MLP-128-64-seed-31415 | 20 | CBF | 2.987651 | 2.194557 | -0.212662 | 0.994682 |
| DNN-residual-MLP-128-64-seed-31415 | 20 | ATT | 0.295425 | 0.191964 | -0.007887 | 0.908201 |

## Findings and interpretation

- The saved H-DELM V2 artifact reproduces its earlier benchmark performance, while newly fitted ELMs are independent calibration fits. These are not the same artifact.
- Learned-representation-only, physics-only, and combined-feature ELMs are compared in `per_model_metrics.csv`; their contribution is unresolved if the combined model does not consistently outperform both component ablations.
- The direct-output ELM is evaluated against residual ELM under identical combined features and a fixed 512-neuron tanh readout.
- Fixed and validation-selected H-DELM correction strengths are tabulated. The calibration validation split selected the reported alpha; test results were not involved.
- Three seeds were run for the combined residual ELM and frozen-backbone residual MLP. Mean/SD are in `multi_seed_results.csv`; no best seed is promoted as the overall result.
- Residual ELM is compared to direct-output ELM with identical feature inputs; both use a fixed ridge lambda of 0.01 and 512 tanh hidden units. This isolates target formulation under that configuration, not across a tuned hyperparameter search.
- Paired Wilcoxon tests are Holm-adjusted across all model/SNR/parameter comparisons. Paired bootstrap intervals and sign-flip permutation tests are included for the saved H-DELM and the three-seed ELM/MLP configurations. A p-value floor is flagged if SciPy underflows to zero.
- Statistical significance should be interpreted with the prior-test reuse limitation above; the comparisons do not constitute an independent replication study.

## Not completed in this run

Kernel residual models, multi-head DNN, additional residual-MLP widths,
detailed feature-group ablations, reduced-PLD retraining, and broader
residual/physiological-range diagnostics remain unrun. Parameter counts,
memory use, and inference latency were not benchmarked. No conclusion is made
about those comparisons.

## Artifact index

- `../results/per_model_metrics.csv`: all model, SNR, parameter, and overall metrics.
- `../results/per_snr_comparison.csv`: direct comparisons against DNN-G.
- `../results/multi_seed_results.csv`: mean and sample SD over the three ELM and MLP seeds.
- `../results/paired_tests_holm.csv`: paired Wilcoxon tests with Holm correction across all listed comparisons.
- `../results/paired_bootstrap_permutation.csv`: paired RMSE bootstrap intervals and paired squared-error permutation tests for primary variants, with Holm correction.
- `../results/predictions_and_truth.npz`: identical test observations, labels, SNRs, and predictions.
- `../results/run_audit.json`: data counts, artifact hashes, seeds, versions, and test reuse disclosure.
