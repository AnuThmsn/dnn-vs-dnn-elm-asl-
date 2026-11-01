import os, json, gc
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT)
import numpy as np, pandas as pd, torch
from experiments.physics import PLDS, TAU, SIG_REF_SC, compute_physical_signals, add_rician_noise, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from experiments.dataset import prepare_and_validate_all_datasets, CACHE_DIR
from experiments.att_diagnostics_m0_calibration.calibrate_m0 import (
    compute_median_reference,
    compute_scale_factor,
    apply_median_scaling,
    noise_sd_from_snr,
)

EXP_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results")
os.makedirs(EXP_DIR, exist_ok=True)
from experiments.train_baseline_A import (
    StandardizedNet,
    train_standardized,
    calc_metrics,
)

# -----------------------------------------------------------------------------
# Helper: generate calibrated splits (train/val/test) using the same ground‑truth
# CBF/ATT and SNR assignments as the original cached data, but optionally
# recompute the input‑space normalization.
# -----------------------------------------------------------------------------

def load_original_split(split_name: str):
    """Return (X, Y, snr) for a cached split.
    Arrays are memory‑mapped (read‑only) to keep RAM low.
    """
    x_path = os.path.join(CACHE_DIR, f"X_{split_name}.npy")
    y_path = os.path.join(CACHE_DIR, f"Y_{split_name}.npy")
    snr_path = os.path.join(CACHE_DIR, f"snr_{split_name}.npy")
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")
    snr = np.load(snr_path, mmap_mode="r")
    return X, Y, snr


def calibrate_split(split_name: str, scale_factor: float, rng_seed: int):
    """Return a calibrated version of a split.
    * clean signals are recomputed from the ground‑truth CBF/ATT values
      (the same as used to generate the original noisy data).
    * the clean signals are scaled by ``scale_factor``.
    * noise is re‑added *with the original SNR values* using the standard
      physics noise model (sigma = SIG_REF_SC / SNR).  This yields the
      ``cal_frozen`` condition.
    """
    X_orig, Y, snr = load_original_split(split_name)
    # Re‑create the clean physical signals from ground‑truth targets
    clean = compute_physical_signals(PLDS, TAU, Y[:, 0], Y[:, 1])
    # Apply median‑based scaling
    clean_scaled = apply_median_scaling(clean, scale_factor)
    # Re‑add Rician noise using the *original* SIG_REF_SC convention
    rng = np.random.default_rng(rng_seed)
    X_cal = add_rician_noise(clean_scaled, snr, rng).astype(np.float32)
    return X_cal, Y, snr


def calibrate_split_with_custom_noise(split_name: str, scale_factor: float, rng_seed: int):
    """Calibration + *explicit* sigma recomputed after scaling.
    The noise standard‑deviation uses the *scaled* reference amplitude:
        sigma = (SIG_REF_SC * scale_factor) / SNR
    This makes the noise level proportional to the actual signal magnitude
    after calibration, differentiating this condition from ``cal_frozen``.
    """
    X_orig, Y, snr = load_original_split(split_name)
    clean = compute_physical_signals(PLDS, TAU, Y[:, 0], Y[:, 1])
    clean_scaled = apply_median_scaling(clean, scale_factor)
    # Compute sigma per the *scaled* reference signal
    sigma_scaled = np.where(np.isinf(snr), 0.0, (SIG_REF_SC * scale_factor) / snr)[:, None]
    rng = np.random.default_rng(rng_seed)
    # Manually add Rician noise using the custom sigma
    D = clean_scaled.shape[1]
    e1 = rng.normal(0.0, 1.0, size=(clean_scaled.shape[0], D)) * sigma_scaled
    e2 = rng.normal(0.0, 1.0, size=(clean_scaled.shape[0], D)) * sigma_scaled
    e3 = rng.normal(0.0, 1.0, size=(clean_scaled.shape[0], D)) * sigma_scaled
    e4 = rng.normal(0.0, 1.0, size=(clean_scaled.shape[0], D)) * sigma_scaled
    mc = np.sqrt((clean_scaled + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((clean_scaled + e3) ** 2 + e4 ** 2)
    X_cal = (mc + ml).astype(np.float32)
    return X_cal, Y, snr


def compute_norm_stats(X: np.ndarray, Y: np.ndarray):
    """Chunked mean/std computation for inputs (4‑dim) and targets (CBF,ATT)."""
    N = X.shape[0]
    chunk = 100_000
    sum_x = np.zeros(4, dtype=np.float64)
    sum_sq_x = np.zeros(4, dtype=np.float64)
    sum_cbf = 0.0
    sum_sq_cbf = 0.0
    sum_att = 0.0
    sum_sq_att = 0.0
    for i in range(0, N, chunk):
        x_c = X[i:i+chunk]
        y_c = Y[i:i+chunk]
        sum_x += x_c.sum(axis=0)
        sum_sq_x += (x_c.astype(np.float64) ** 2).sum(axis=0)
        sum_cbf += y_c[:, 0].sum()
        sum_sq_cbf += (y_c[:, 0] ** 2).sum()
        sum_att += y_c[:, 1].sum()
        sum_sq_att += (y_c[:, 1] ** 2).sum()
    X_mean = (sum_x / N).astype(np.float32).reshape(1, 4)
    X_std = np.sqrt((sum_sq_x / N) - (X_mean.squeeze() ** 2)).astype(np.float32).reshape(1, 4)
    cbf_mean = float(sum_cbf / N)
    cbf_std = float(np.sqrt((sum_sq_cbf / N) - (cbf_mean ** 2)))
    att_mean = float(sum_att / N)
    att_std = float(np.sqrt((sum_sq_att / N) - (att_mean ** 2)))
    return X_mean, X_std, cbf_mean, cbf_std, att_mean, att_std


def train_and_evaluate(condition: str, X_tr, Y_tr, X_val, Y_val, X_test, Y_test, norm_stats, seed: int, pre_normalized: bool = False):
    """Train ATT & CBF nets for a given condition and save metrics/predictions.
    ``norm_stats`` is a tuple produced by ``compute_norm_stats`` (or the
    baseline stats).  ``seed`` controls torch randomness only.
    """
    pred_path = os.path.join(EXP_DIR, f"{condition}_seed{seed}_predictions.npz")
    metric_path = os.path.join(EXP_DIR, f"{condition}_seed{seed}_overall_metrics.csv")
    if os.path.exists(pred_path) and os.path.exists(metric_path):
        try:
            with np.load(pred_path) as p:
                if "cbf_test_pred" in p and len(p["cbf_test_pred"]) == len(Y_test) and np.all(np.isfinite(p["cbf_test_pred"])):
                    print(f"[{condition} | Seed {seed}] Valid outputs already exist at {pred_path}. Skipping training.", flush=True)
                    return pred_path
        except Exception as e:
            print(f"[{condition} | Seed {seed}] Existing file {pred_path} invalid ({e}), retraining.", flush=True)

    torch.manual_seed(seed)
    X_mean, X_std, cbf_mean, cbf_std, att_mean, att_std = norm_stats

    # Normalise inputs if not already done
    if pre_normalized:
        X_tr_n, X_val_n, X_test_n = X_tr, X_val, X_test
    else:
        X_tr_n = ((X_tr - X_mean) / X_std).astype(np.float32)
        X_val_n = ((X_val - X_mean) / X_std).astype(np.float32)
        X_test_n = ((X_test - X_mean) / X_std).astype(np.float32)

    INPUT_DIM = len(PLDS)
    ATTnet = StandardizedNet(INPUT_DIM, n_hidden_layers=4, n_neurons=64,
                             y_mean=att_mean, y_std=att_std,
                             y_min=ATT_MIN, y_max=ATT_MAX).to('cpu')
    CBFnet = StandardizedNet(INPUT_DIM, n_hidden_layers=5, n_neurons=128,
                             y_mean=cbf_mean, y_std=cbf_std,
                             y_min=CBF_MIN, y_max=CBF_MAX).to('cpu')

    att_ckpt_path = os.path.join(EXP_DIR, f"{condition}_seed{seed}_ATT_best.pth")
    cbf_ckpt_path = os.path.join(EXP_DIR, f"{condition}_seed{seed}_CBF_best.pth")
    recovery_ckpt_path = os.path.join(EXP_DIR, f"{condition}_seed{seed}_recovery.pth")

    print(f"\n[{condition} | Seed {seed}] Starting training ATT net...", flush=True)
    # Train ATT net and capture history
    ATTnet, att_hist, att_time = train_standardized(
        ATTnet, X_tr_n, Y_tr[:, 1], X_val_n, Y_val[:, 1],
        lr=1e-3, max_epochs=200, patience=10,
        log_prefix=f"[{condition} | Seed {seed} | ATT] ",
        best_ckpt_path=att_ckpt_path,
        recovery_ckpt_path=recovery_ckpt_path)

    # Save ATT training history
    att_hist_path = os.path.join(EXP_DIR, f"{condition}_seed{seed}_ATT_history.csv")
    pd.DataFrame(att_hist).to_csv(att_hist_path, index=False)
    print(f"[{condition} | Seed {seed}] Saved ATT training history to {att_hist_path}", flush=True)

    print(f"\n[{condition} | Seed {seed}] Starting training CBF net...", flush=True)
    # Train CBF net and capture history
    CBFnet, cbf_hist, cbf_time = train_standardized(
        CBFnet, X_tr_n, Y_tr[:, 0], X_val_n, Y_val[:, 0],
        lr=1e-3, max_epochs=200, patience=10,
        log_prefix=f"[{condition} | Seed {seed} | CBF] ",
        best_ckpt_path=cbf_ckpt_path,
        recovery_ckpt_path=recovery_ckpt_path)

    # Save CBF training history
    cbf_hist_path = os.path.join(EXP_DIR, f"{condition}_seed{seed}_CBF_history.csv")
    pd.DataFrame(cbf_hist).to_csv(cbf_hist_path, index=False)
    print(f"[{condition} | Seed {seed}] Saved CBF training history to {cbf_hist_path}", flush=True)

    # Save final best-validation and recovery checkpoints
    torch.save(ATTnet.state_dict(), att_ckpt_path)
    torch.save(CBFnet.state_dict(), cbf_ckpt_path)
    torch.save({"ATTnet": ATTnet.state_dict(), "CBFnet": CBFnet.state_dict()}, recovery_ckpt_path)
    print(f"[{condition} | Seed {seed}] Saved checkpoints: {att_ckpt_path}, {cbf_ckpt_path}, {recovery_ckpt_path}", flush=True)

    # Predictions on test set
    cbf_pred = CBFnet.predict_physical(X_test_n)
    att_pred = ATTnet.predict_physical(X_test_n)

    # Save predictions for later metric recomputation
    np.savez(pred_path,
             cbf_test_pred=cbf_pred,
             att_test_pred=att_pred,
             cbf_true=Y_test[:, 0],
             att_true=Y_test[:, 1])
    print(f"[{condition} | Seed {seed}] Saved predictions to {pred_path}", flush=True)

    # Compute baseline metrics (overall test set) – stored for reference
    # Detailed SNR‑wise metrics are computed later from the saved arrays.
    overall = pd.DataFrame([
        {"Target": "CBF", **calc_metrics(Y_test[:, 0], cbf_pred)},
        {"Target": "ATT", **calc_metrics(Y_test[:, 1], att_pred)}
    ])
    overall.to_csv(metric_path, index=False)
    print(f"[{condition} | Seed {seed}] Saved overall metrics to {metric_path}", flush=True)
    return pred_path


def evaluate_snr_metrics(pred_path: str):
    """Load a ``*_predictions.npz`` file and compute metrics for SNR=10,15,20.
    Returns a DataFrame with rows for each target and SNR.
    """
    data = np.load(pred_path)
    cbf_true = data["cbf_true"]
    att_true = data["att_true"]
    cbf_pred = data["cbf_test_pred"]
    att_pred = data["att_test_pred"]
    # The SNR values that correspond to each sample are stored in the
    # original cached ``snr_test.npy`` (identical across conditions).
    snr_test = np.load(os.path.join(CACHE_DIR, "snr_test.npy"))
    rows = []
    for target_name, true, pred in [("CBF", cbf_true, cbf_pred), ("ATT", att_true, att_pred)]:
        for snr_val in (10.0, 15.0, 20.0):
            mask = snr_test == snr_val
            if mask.sum() == 0:
                continue
            metrics = calc_metrics(true[mask], pred[mask])
            rows.append({
                "Condition": os.path.basename(pred_path).split('_seed')[0],
                "Seed": int(os.path.basename(pred_path).split('seed')[1].split('_')[0]),
                "Target": target_name,
                "SNR": snr_val,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "Bias": metrics.get("Bias", np.nan),
                "CCC": metrics["CCC"],
                "NormRMSE": metrics.get("NormRMSE", np.nan)
            })
    return pd.DataFrame(rows)


def main():
    # ---------------------------------------------------------------------
    # 0. Prepare original cached datasets (train/val/test) and baseline stats
    # ---------------------------------------------------------------------
    data = prepare_and_validate_all_datasets()
    # Original baseline stats (frozen)
    baseline_stats = (
        data["X_mean"], data["X_std"],
        data["cbf_mean"], data["cbf_std"],
        data["att_mean"], data["att_std"]
    )

    # ---------------------------------------------------------------------
    # 1. Load clean training signals to compute the median reference
    # ---------------------------------------------------------------------
    clean_train = compute_physical_signals(PLDS, TAU,
                                          np.load(data["y_tr_path"], mmap_mode='r')[:, 0],
                                          np.load(data["y_tr_path"], mmap_mode='r')[:, 1])
    median_ref = compute_median_reference(clean_train)
    scale_factor = compute_scale_factor(median_ref)

    # ---------------------------------------------------------------------
    # 2. Create calibrated splits (same ground‑truth, same SNR)
    # ---------------------------------------------------------------------
    X_tr_cal, Y_tr_cal, snr_tr_cal = calibrate_split("train", scale_factor, rng_seed=42)
    X_val_cal, Y_val_cal, snr_val_cal = calibrate_split("val", scale_factor, rng_seed=99)
    X_test_cal, Y_test_cal, snr_test_cal = calibrate_split("test", scale_factor, rng_seed=7)

    # Calibrated + custom‑noise split (different sigma)
    X_tr_cal_noise, Y_tr_cal_noise, _ = calibrate_split_with_custom_noise("train", scale_factor, rng_seed=42)
    X_val_cal_noise, Y_val_cal_noise, _ = calibrate_split_with_custom_noise("val", scale_factor, rng_seed=99)
    X_test_cal_noise, Y_test_cal_noise, _ = calibrate_split_with_custom_noise("test", scale_factor, rng_seed=7)

    # ---------------------------------------------------------------------
    # 3. Normalisation for train‑only condition (re‑compute from calibrated data)
    # ---------------------------------------------------------------------
    norm_cal_trainonly = compute_norm_stats(X_tr_cal, Y_tr_cal)

    # ---------------------------------------------------------------------
    # 4. Pre-normalize inputs for each condition to eliminate redundant 1M-sample work
    # ---------------------------------------------------------------------
    print("Pre-normalizing datasets across conditions to reduce redundant computation...", flush=True)
    X_mean_b, X_std_b = baseline_stats[0], baseline_stats[1]

    # Baseline
    X_tr_base = np.load(data["x_tr_path"], mmap_mode='r')
    X_val_base = np.load(data["x_val_path"], mmap_mode='r')
    X_test_base = np.load(data["x_test_path"], mmap_mode='r')
    Y_tr_base = np.load(data["y_tr_path"], mmap_mode='r')
    Y_val_base = np.load(data["y_val_path"], mmap_mode='r')
    Y_test_base = np.load(data["y_test_path"], mmap_mode='r')

    X_tr_base_n = ((X_tr_base - X_mean_b) / X_std_b).astype(np.float32)
    X_val_base_n = ((X_val_base - X_mean_b) / X_std_b).astype(np.float32)
    X_test_base_n = ((X_test_base - X_mean_b) / X_std_b).astype(np.float32)

    # Cal-frozen (uses baseline stats)
    X_tr_cal_frozen_n = ((X_tr_cal - X_mean_b) / X_std_b).astype(np.float32)
    X_val_cal_frozen_n = ((X_val_cal - X_mean_b) / X_std_b).astype(np.float32)
    X_test_cal_frozen_n = ((X_test_cal - X_mean_b) / X_std_b).astype(np.float32)

    # Cal-trainonly (uses norm_cal_trainonly)
    X_mean_to, X_std_to = norm_cal_trainonly[0], norm_cal_trainonly[1]
    X_tr_cal_to_n = ((X_tr_cal - X_mean_to) / X_std_to).astype(np.float32)
    X_val_cal_to_n = ((X_val_cal - X_mean_to) / X_std_to).astype(np.float32)
    X_test_cal_to_n = ((X_test_cal - X_mean_to) / X_std_to).astype(np.float32)

    # Cal-noise (uses baseline stats)
    X_tr_cal_noise_n = ((X_tr_cal_noise - X_mean_b) / X_std_b).astype(np.float32)
    X_val_cal_noise_n = ((X_val_cal_noise - X_mean_b) / X_std_b).astype(np.float32)
    X_test_cal_noise_n = ((X_test_cal_noise - X_mean_b) / X_std_b).astype(np.float32)
    print("Pre-normalization complete.", flush=True)

    # ---------------------------------------------------------------------
    # 5. Run experiments for seeds 42, 43, 44 (resuming missing runs only)
    # ---------------------------------------------------------------------
    seeds = [42, 43, 44]
    pred_paths = []
    for seed in seeds:
        print(f"\n==================================================", flush=True)
        print(f"=== PROCESSING SEED {seed} ===", flush=True)
        print(f"==================================================", flush=True)
        # Baseline (frozen stats, original data)
        pred_paths.append(
            train_and_evaluate(
                f"baseline_seed{seed}",
                X_tr_base_n, Y_tr_base,
                X_val_base_n, Y_val_base,
                X_test_base_n, Y_test_base,
                baseline_stats,
                seed,
                pre_normalized=True,
            )
        )
        # Cal‑Frozen (median scaling, frozen baseline stats)
        pred_paths.append(
            train_and_evaluate(
                f"cal_frozen_seed{seed}",
                X_tr_cal_frozen_n, Y_tr_cal,
                X_val_cal_frozen_n, Y_val_cal,
                X_test_cal_frozen_n, Y_test_cal,
                baseline_stats,
                seed,
                pre_normalized=True,
            )
        )
        # Cal‑TrainOnly (median scaling, stats recomputed on calibrated train)
        pred_paths.append(
            train_and_evaluate(
                f"cal_trainonly_seed{seed}",
                X_tr_cal_to_n, Y_tr_cal,
                X_val_cal_to_n, Y_val_cal,
                X_test_cal_to_n, Y_test_cal,
                norm_cal_trainonly,
                seed,
                pre_normalized=True,
            )
        )
        # Cal‑Noise (median scaling, custom sigma after scaling)
        pred_paths.append(
            train_and_evaluate(
                f"cal_noise_seed{seed}",
                X_tr_cal_noise_n, Y_tr_cal_noise,
                X_val_cal_noise_n, Y_val_cal_noise,
                X_test_cal_noise_n, Y_test_cal_noise,
                baseline_stats,
                seed,
                pre_normalized=True,
            )
        )
    # ---------------------------------------------------------------------
    # 5. Aggregate SNR‑wise metrics across all conditions and seeds
    # ---------------------------------------------------------------------
    all_dfs = []
    for p in pred_paths:
        all_dfs.append(evaluate_snr_metrics(p))
    agg = pd.concat(all_dfs, ignore_index=True)
    agg_path = os.path.join(EXP_DIR, "snr_metrics_aggregated.csv")
    agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated SNR metrics to {agg_path}")

    # ---------------------------------------------------------------------
    # 6. Evaluate original baseline checkpoints on the *original* locked test set
    # ---------------------------------------------------------------------
    # Load checkpoints (they live under experiments/baseline_A)
    baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'baseline_A'))
    att_ckpt = os.path.join(baseline_dir, "ATTnet_G_baselineA.pth")
    cbf_ckpt = os.path.join(baseline_dir, "CBFnet_G_baselineA.pth")
    # Load normalization stats used for the original baseline
    norm_path = os.path.join(baseline_dir, "normalization_stats.npz")
    norm = np.load(norm_path)
    X_mean = norm["X_mean"]
    X_std = norm["X_std"]
    cbf_mean = float(norm["cbf_mean"])
    cbf_std = float(norm["cbf_std"])
    att_mean = float(norm["att_mean"])
    att_std = float(norm["att_std"])
    # Prepare test inputs
    X_test = np.load(data["x_test_path"], mmap_mode='r')
    Y_test = np.load(data["y_test_path"], mmap_mode='r')
    X_test_n = ((X_test - X_mean) / X_std).astype(np.float32)
    # Build nets and load weights
    INPUT_DIM = len(PLDS)
    ATTnet = StandardizedNet(INPUT_DIM, n_hidden_layers=4, n_neurons=64,
                             y_mean=att_mean, y_std=att_std,
                             y_min=ATT_MIN, y_max=ATT_MAX).to('cpu')
    CBFnet = StandardizedNet(INPUT_DIM, n_hidden_layers=5, n_neurons=128,
                             y_mean=cbf_mean, y_std=cbf_std,
                             y_min=CBF_MIN, y_max=CBF_MAX).to('cpu')
    ATTnet.load_state_dict(torch.load(att_ckpt, map_location='cpu'))
    CBFnet.load_state_dict(torch.load(cbf_ckpt, map_location='cpu'))
    # Predictions
    att_pred = ATTnet.predict_physical(X_test_n)
    cbf_pred = CBFnet.predict_physical(X_test_n)
    baseline_metrics = pd.DataFrame([
        {"Target": "CBF", **calc_metrics(Y_test[:, 0], cbf_pred)},
        {"Target": "ATT", **calc_metrics(Y_test[:, 1], att_pred)}
    ])
    baseline_metrics_path = os.path.join(EXP_DIR, "original_baseline_test_metrics.csv")
    baseline_metrics.to_csv(baseline_metrics_path, index=False)
    print(f"Baseline metrics (original checkpoints) saved to {baseline_metrics_path}")

if __name__ == "__main__":
    main()

