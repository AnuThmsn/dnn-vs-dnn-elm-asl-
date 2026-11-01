"""
physics.py
Authoritative Buxton General Kinetic Model (GKM) implementation and Dual-Channel Rician Noise Model.
Source: dnn-elm-noise (7).ipynb (Cells 3 & 4) and asl-dnn-papermatched-v2-noise-density.ipynb (Cells 3 & 4).
"""

import numpy as np

# Physical & Acquisition Constants (Strictly Frozen)
PLDS = np.array([1.525, 2.025, 2.525, 3.025], dtype=np.float64)  # seconds
TAU = 1.8                                                        # seconds (labeling duration)
T1T = 1.2                                                        # seconds (tissue T1)
T1A = 1.66                                                       # seconds (blood T1)
ALPHA = 0.85                                                     # inversion efficiency
BETA = 0.75                                                      # transit attenuation
LAMBDA = 0.9                                                     # blood-brain partition coefficient (mL/g)
SCALE = 100_000.0                                                # global signal scale factor

# Physiological Parameter Bounds
CBF_MIN = 0.0                                                    # mL / (100g * min)
CBF_MAX = 100.0
ATT_MIN = 0.5                                                    # seconds
ATT_MAX = 3.0

# Reference Signal Point (CBF=50, ATT=1.6, PLD=2.0, LD=1.8)
REF_CBF = 50.0
REF_ATT = 1.6
REF_PLD = 2.0
REF_LD = 1.8

# SNR Configuration
SNR_VALUES = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, np.inf]


def asl_signal(pld, tau_val, f_val, att_val):
    """
    Standard Buxton General Kinetic Model (GKM) for pCASL.
    Vectorized over pld, f_val, att_val.
    """
    pld = np.asarray(pld, dtype=np.float64)
    att_val = np.asarray(att_val, dtype=np.float64)
    f_val = np.asarray(f_val, dtype=np.float64)

    t_delay = np.maximum(pld - att_val, 0.0)
    t_end = np.maximum(tau_val + pld - att_val, 0.0)
    decay = np.exp(-t_delay / T1T) - np.exp(-t_end / T1T)

    coeff = 2.0 * ALPHA * BETA * T1T * (1.0 / LAMBDA) * f_val * np.exp(-att_val / T1A)
    return coeff * decay


def cbf_to_f(cbf):
    """Convert CBF [mL / (100g * min)] to physical perfusion rate f [s^-1]."""
    return np.asarray(cbf, dtype=np.float64) / (6000.0 * LAMBDA)


def compute_physical_signals(plds, tau_val, cbf, att):
    """
    Compute clean 1x scaled ASL signals.
    Inputs:
        plds: (D,) array of post-labeling delays
        cbf: (N,) or scalar CBF values
        att: (N,) or scalar ATT values
    Returns:
        signals: (N, D) scaled signals in physical units (1x scale)
    """
    cbf = np.atleast_1d(cbf).reshape(-1, 1)
    att = np.atleast_1d(att).reshape(-1, 1)
    plds = np.atleast_1d(plds).reshape(1, -1)

    f_val = cbf_to_f(cbf)
    sig = asl_signal(plds, tau_val, f_val, att) * SCALE
    return sig


# Reference Signal Constant
_f_ref = cbf_to_f(REF_CBF)
SIG_REF_SC = float(np.abs(asl_signal(REF_PLD, REF_LD, _f_ref, REF_ATT)) * SCALE)


def add_rician_noise(clean_signals, snr, rng):
    """
    Dual-channel quadrature Rician magnitude noise simulation.
    Inputs:
        clean_signals: (N, 4) clean 1x signals S
        snr: scalar or (N,) array of SNR values
        rng: np.random.Generator instance
    Returns:
        X: (N, 4) observed signals (Mc + Ml). Clean baseline is 2 * S.
    """
    clean_signals = np.asarray(clean_signals, dtype=np.float64)
    N, D = clean_signals.shape

    if np.isscalar(snr):
        snr_arr = np.full(N, snr, dtype=np.float64)
    else:
        snr_arr = np.asarray(snr, dtype=np.float64)

    # Calculate noise SD: sigma = SIG_REF_SC / SNR
    sd_arr = np.where(np.isinf(snr_arr), 0.0, SIG_REF_SC / snr_arr)[:, None]

    e1 = rng.normal(0.0, 1.0, size=(N, D)) * sd_arr
    e2 = rng.normal(0.0, 1.0, size=(N, D)) * sd_arr
    e3 = rng.normal(0.0, 1.0, size=(N, D)) * sd_arr
    e4 = rng.normal(0.0, 1.0, size=(N, D)) * sd_arr

    mc = np.sqrt((clean_signals + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((clean_signals + e3) ** 2 + e4 ** 2)
    return (mc + ml).astype(np.float32)

