# Objective 1 — H-DELM architecture verification

**Status: NOT VERIFIED against the reference paper.**

## Reference-paper availability

No source paper describing H-DELM was found in the workspace. In particular, no
paper PDF, LaTeX source, or bibliographic reference that can be checked against
the implementation was available. The string titled *A Residual Hybrid
DNN-ELM Model for CBF and ATT Estimation in Multi-Delay ASL MRI* in
`experiments/paper_reproduction_snr10_15_20_v2/config/experiment_config_v2.json`
is experiment configuration metadata, not the paper itself or a verifiable
citation. The uploaded Mastropietro et al. (2022) IVIM paper is not used as an
H-DELM reference.

Consequently, no paper specifications are inferred or invented. The table
records what the code does and whether a paper-level match can be established.

## Specification comparison

| Paper specification | Actual implementation | Verification status |
|---|---|---|
| DNN input count and feature definition: **not available** | `DNN_G_Net` accepts four normalized PLD measurements (`input_dim=4`). | Implementation verified; paper match not verifiable. (`models/dnn_g.py`, `DNN_G_Net.__init__`, lines 25–42.) |
| DNN hidden-layer count/width: **not available** | V1 H-DELM pipeline creates separate CBF and ATT `DNN_G_Net`s, each with 4 hidden layers × 64 units. `experiments/baseline_A` is a different model family: CBF 5 × 128 and ATT 4 × 64. | Implementation verified; paper match not verifiable. Repository has distinct configurations. (`run_reproduction_pipeline.py`, lines 114–131; `train_baseline_A.py`, lines 230–234.) |
| DNN activation: **not available** | ELU after each of the four hidden linear layers. | Implementation verified; paper match not verifiable. (`models/dnn_g.py`, lines 35–39.) |
| DNN weight/bias initialization: **not available** | Kaiming-normal weights with `nonlinearity="relu"` and zero biases, including the output linear layer. | Implementation verified; paper match not verifiable. (`models/dnn_g.py`, lines 44–48.) |
| DNN output head and bounds: **not available** | One linear logit; sigmoid maps to `[0,1]`, then affine rescaling gives CBF `[0,100]` or ATT `[0.5,3.0]`. | Implementation verified; paper match not verifiable. (`models/dnn_g.py`, lines 42–58, 60–75.) |
| Input/target normalization: **not available** | Input mean/std are calculated from DNN training inputs only and used for both train and validation. Targets use min-max physical-bound normalization, not z-score standardization: `(y-y_min)/(y_max-y_min)`. | Implementation verified; paper match not verifiable. (`run_reproduction_pipeline.py`, lines 89–103; `models/dnn_g.py`, lines 102–115.) |
| Loss and optimizer: **not available** | MSE between sigmoid output and min-max target; Adam at `1e-3`, batch size 512, up to 200 epochs, patience 10, gradient norm clipped at 1.0; best validation weights retained. | Implementation verified; paper match not verifiable. (`models/dnn_g.py`, lines 94–118, 135–171.) |
| ELM existence/role: **not available** | `ResidualHDELMV2` is an actual ELM-style random-hidden-layer ridge readout. It learns `r = y_true - y_DNN` and returns `clip(y_DNN + r_hat, y_min, y_max)`. | Residual ELM is implemented; paper match not verifiable. (`models/residual_hdelm_v2.py`, `ResidualHDELMV2.fit`, lines 225–271; `predict_physical`, lines 304–312.) |
| ELM inputs: **not available** | Concatenates 64 DNN penultimate activations with 15 raw-signal-derived features: 4 signals, 3 PLD slopes, 3 adjacent ratios, mean, variance, min, max, and trapezoidal AUC; total 79. | Implementation verified; paper match not verifiable. (`models/residual_hdelm_v2.py`, `extract_15_physics_features`, lines 40–81; `extract_combined_features`, lines 119–127.) |
| ELM hidden width/activation: **not available** | 512 hidden units; `tanh` activation. | Implementation verified; paper match not verifiable. (`models/residual_hdelm_v2.py`, lines 83–99, 264–271.) |
| Fixed random hidden weights/biases: **not available** | NumPy RNG seed 42 initializes `W` and `b` once from uniform `[-1/sqrt(79), +1/sqrt(79)]`; these are not gradient-trained. Saved artifacts contain the initialized `W` and `b`. | Implementation verified; paper match not verifiable. (`models/residual_hdelm_v2.py`, lines 111–117, 314–354.) |
| Ridge readout and regularization: **not available** | Output `beta` is solved analytically from accumulated `HᵀH` and `Hᵀr` using `(HᵀH + λI)⁻¹Hᵀr`; final artifacts store `λ=0.01`. The calibration sweep tests `[1e-4, 1e-3, 1e-2, 1e-1, 1.0]`, but its lowest validation RMSE is at `1.0` for both CBF and ATT; the driver still fits the final artifacts with hard-coded `0.01`. | Implementation verified; paper match not verifiable. Internal pipeline discrepancy: the selected sweep value is not used for final fitting. (`models/residual_hdelm_v2.py`, lines 130–221, 264–271, 330–341; `run_reproduction_pipeline_v2.py`, lines 129–168.) |
| Residual target and final combination: **not available** | Calibration residual is explicitly `Y_cal_phys - y_dnn_cal`; final prediction is DNN output plus predicted ELM residual, clipped to physiological bounds. | Implementation verified; paper match not verifiable. (`models/residual_hdelm_v2.py`, lines 245–271, 304–312.) |

## Findings and scope

- The tested V2 model is a real residual ELM implementation, not a weighted
  average mislabeled as an ELM: it uses a fixed random hidden projection and a
  ridge-regression readout, and its saved artifact contains `W`, `b`, `beta`,
  feature mean/std, and metadata.
- V2 uses 100,000 calibration observations, of which 80,000 are used for
  calibration fitting and 20,000 for calibration validation/hyperparameter
  selection. It does not update the frozen DNN. The old V1 `ResidualHDELM` instead
  fits residuals on the DNN's training data; V2 is the independently calibrated
  variant used for the final test.
- The V2 configuration documents a proposed architecture, but that is not
  evidence of what the missing reference paper specifies. No claim of paper
  reproduction can therefore be made; the code is an implementation/adaptation
  whose fidelity to the paper remains unknown.
- The V2 driver records a lambda sensitivity sweep but uses `ridge_lambda=1e-2`
  when fitting the saved final ELMs. The stored validation sweep CSVs show the
  lowest validation RMSE at `lambda=1.0` for both targets; therefore the final
  model does not use the best value found by that sweep. This behavior was
  preserved for the requested evaluation; no parameter was changed.
