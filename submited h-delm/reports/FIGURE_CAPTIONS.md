# Figure captions

1. **Implemented H-DELM V2 architecture.** Four normalized PLD samples feed
   the frozen DNN-G and the signal-feature branch. The learned 64-dimensional
   DNN representation and 15 signal-derived values are concatenated and passed
   through a fixed 512-unit `tanh` ELM with ridge readout. Its predicted
   residual is added to the DNN prediction and clipped to physical bounds.
2. **DNN-G training and validation losses.** Actual normalized MSE histories
   for CBF and ATT, plotted by epoch without smoothing.
3. **CBF RMSE by SNR.** DNN-G and H-DELM V2 CBF RMSE on the same 10,000
   simulated samples at each SNR.
4. **ATT RMSE by SNR.** DNN-G and H-DELM V2 ATT RMSE on the same 10,000
   simulated samples at each SNR.
5. **CBF estimates versus ground truth.** DNN-G and H-DELM V2 predictions
   versus true CBF for the independent simulated test cohort, with identity
   and least-squares lines and full-cohort RMSE, MAE, CCC, and sample count.
6. **ATT estimates versus ground truth.** Same as Figure 5 for ATT.
7. **Bland–Altman simulated-data agreement.** Differences (estimate minus
   truth) versus paired means, with bias and 95% limits of agreement for CBF
   and ATT. These are simulated parameter estimates, not clinical measurements.
8. **Prediction error distributions by SNR.** Overlaid DNN-G and H-DELM V2
   error densities at SNR 10, 15, and 20, with zero-error reference.
9. **ELM residual correction diagnostics.** DNN residual versus truth,
   predicted versus actual residual, errors before versus after correction,
   and overall error distributions on identical test observations.
10. **Relative RMSE improvement.** Percent RMSE improvement of H-DELM V2
    relative to DNN-G; positive values favor H-DELM and negative values mean
    worse RMSE.
11. **Paired statistical comparison.** Paired-bootstrap 95% intervals for
    DNN-G RMSE minus H-DELM RMSE, with paired Wilcoxon p-values adjusted by
    Holm's method across six comparisons. Positive differences favor H-DELM.

