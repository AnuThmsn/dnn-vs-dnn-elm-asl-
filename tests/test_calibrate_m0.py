import numpy as np
from experiments.physics import SIG_REF_SC
from experiments.att_diagnostics_m0_calibration.calibrate_m0 import (
    compute_median_reference,
    compute_scale_factor,
    apply_median_scaling,
    noise_sd_from_snr,
)

def test_scaling_makes_median_equal_reference():
    # generate synthetic clean signals
    rng = np.random.default_rng(123)
    clean = rng.normal(loc=0.0, scale=1.0, size=(1000, 4)) * 5000  # arbitrary scale
    median_ref = compute_median_reference(clean)
    scale = compute_scale_factor(median_ref)
    scaled = apply_median_scaling(clean, scale)
    new_median = compute_median_reference(scaled)
    assert np.isclose(new_median, SIG_REF_SC, atol=1e-6), f"Median after scaling {new_median} != SIG_REF_SC {SIG_REF_SC}"

def test_noise_sd_matches_physics():
    snr_vals = np.array([5.0, 10.0, np.inf, 20.0])
    sd = noise_sd_from_snr(snr_vals)
    expected = np.where(np.isinf(snr_vals), 0.0, SIG_REF_SC / snr_vals)
    assert np.allclose(sd, expected), "noise_sd_from_snr does not match physics definition"

if __name__ == "__main__":
    test_scaling_makes_median_equal_reference()
    test_noise_sd_matches_physics()
    print("All tests passed.")

