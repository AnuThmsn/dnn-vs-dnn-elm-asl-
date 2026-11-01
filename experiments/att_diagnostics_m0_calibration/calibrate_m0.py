"""calibrate_m0.py
Utility functions for the median‑based M₀ calibration experiment.
The original draft scaled signals by `median_ref / SIG_REF_SC`, which
does **not** guarantee that the median of the scaled signals equals the
reference amplitude `SIG_REF_SC`.  The corrected implementation uses the
inverse factor so that:

    median(|scaled_signal|) == SIG_REF_SC

This matches the intention of a *median reference* calibration: after
scaling, the median absolute signal amplitude should be the reference
signal magnitude used in the noise model.

The module provides:
* `compute_median_reference(signals)`: median of absolute values.
* `compute_scale_factor(median_ref)`: returns `SIG_REF_SC / median_ref`.
* `apply_median_scaling(signals, scale)`: multiplies signals by the
  scale factor.
* `noise_sd_from_snr(snr)`: helper returning sigma = SIG_REF_SC / snr.
"""
import numpy as np
from experiments.physics import SIG_REF_SC


def compute_median_reference(signals: np.ndarray) -> float:
    """Return the median of the absolute signal values.

    Parameters
    ----------
    signals: np.ndarray
        Array of shape (N, D) containing clean ASL signals.
    """
    return float(np.median(np.abs(signals)))


def compute_scale_factor(median_ref: float) -> float:
    """Compute the scaling factor so that the median becomes `SIG_REF_SC`.

    The factor is defined as ``SIG_REF_SC / median_ref``.
    """
    if median_ref == 0:
        raise ValueError("Median reference is zero; cannot compute scale factor.")
    return SIG_REF_SC / median_ref


def apply_median_scaling(signals: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale the clean signals by the provided factor.

    The operation is element‑wise multiplication and returns a new array.
    """
    return signals * scale_factor


def noise_sd_from_snr(snr: float | np.ndarray) -> np.ndarray:
    """Return the noise standard‑deviation corresponding to the given SNR.

    Mirrors the implementation in ``add_rician_noise``.
    """
    snr_arr = np.asarray(snr, dtype=np.float64)
    # ``np.isinf`` handles infinite SNR (zero noise).
    sd = np.where(np.isinf(snr_arr), 0.0, SIG_REF_SC / snr_arr)
    return sd

# Simple sanity check when the module is executed directly.
if __name__ == "__main__":
    # Dummy clean signals for quick demo.
    dummy = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    median_ref = compute_median_reference(dummy)
    scale = compute_scale_factor(median_ref)
    scaled = apply_median_scaling(dummy, scale)
    print(f"median_ref={median_ref:.4f}, scale={scale:.4f}, new median={compute_median_reference(scaled):.4f}")

