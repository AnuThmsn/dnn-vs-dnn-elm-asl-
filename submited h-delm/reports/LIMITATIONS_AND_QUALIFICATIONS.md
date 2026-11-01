# Limitations and qualifications

1. **Reference paper unavailable.** No source publication describing the
   proposed H-DELM was found, so comparison with paper specifications is
   impossible and no claim of exact reproduction is made.
2. **DNN training cohort provenance.** The V1 generator and training pipeline
   specify 1,000,000 observations, and saved epoch histories/checkpoints are
   consistent with execution. The original arrays were generated in memory
   and were not retained with checksums or a checkpoint run manifest.
3. **Distinct experiment families.** `experiments/baseline_A` is not the
   4×64 DNN-G backbone in the evaluated H-DELM V2 run. The latter is compared
   to its own DNN-G checkpoints.
4. **Ridge parameter discrepancy.** Calibration validation tables favor
   λ=1.0 for both outputs, but the saved final ELM artifacts record λ=0.01.
   The test uses those existing artifacts unchanged.
5. **Simulation scope.** Metrics and agreement plots describe generated
   Buxton-model data with simulated Rician noise, not clinical data.
6. **Statistical scope.** Wilcoxon tests are applied to paired squared errors,
   followed by Holm adjustment over six comparisons. Bootstrap intervals
   separately quantify the paired RMSE improvement distribution.
7. **Sampled scatter plots.** Prediction scatter panels display a seeded
   8,000-point subset for legibility; annotations and all numeric metrics use
   all 30,000 test observations.
8. **Bland–Altman interpretation.** Agreement limits are calculated across
   the entire simulated test cohort and may obscure SNR-specific behavior.

