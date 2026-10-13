"""
feature_engineering.py
Permitted signal-derived feature representations for ASL-MRI parameter estimation.

STRICT LEAKAGE-FREE RULES:
- All features are calculated ONLY from observed 4-PLD signals and known acquisition settings.
- ZERO ground-truth CBF, ATT, hidden test labels, or quantities derived from them.
- All scalers (mean, std) are fitted exclusively on the training partition.
- Non-redundancy is enforced: Relative contrasts are used for signal differences to prevent
  exact linear rank deficiency.
"""

import numpy as np

PLDS = np.array([1.525, 2.025, 2.525, 3.025], dtype=np.float32)
TAU = 1.8

def extract_raw_features(X_raw):
    """Representation B: 4 raw ASL measurement signals."""
    return X_raw.astype(np.float32)

def extract_normalized_diff_features(X_raw, X_mean, X_std, eps=1.0):
    """
    Representation C: Carefully defined normalized signal measurements and non-linear contrast differences:
      1-4: Standardized signals: (x_i - mu_i) / sigma_i
      5-7: Normalized relative temporal contrasts: (x_{i+1} - x_i) / (x_{i+1} + x_i + eps)
    Provides temporal slope information without rank-1 linear redundancy.
    """
    X_n = ((X_raw - X_mean) / X_std).astype(np.float32)

    c21 = (X_raw[:, 1] - X_raw[:, 0]) / (X_raw[:, 1] + X_raw[:, 0] + eps)
    c32 = (X_raw[:, 2] - X_raw[:, 1]) / (X_raw[:, 2] + X_raw[:, 1] + eps)
    c43 = (X_raw[:, 3] - X_raw[:, 2]) / (X_raw[:, 3] + X_raw[:, 2] + eps)

    feats = np.column_stack([X_n, c21, c32, c43])
    return feats.astype(np.float32)

def extract_physics_features(X_raw, eps=1e-3):
    """
    Representation D: Physically meaningful features derived from the Buxton pCASL forward model:
      1-3: Inter-PLD temporal slopes: (x_{i+1} - x_i) / (PLD_{i+1} - PLD_i)
      4-6: Successive decay ratios: x_{i+1} / (x_i + eps)
      7:   Apparent exponential relaxation rate: -ln(max(x4, eps) / max(x1, eps)) / (PLD4 - PLD1)
      8:   Normalized bolus trapezoidal AUC: AUC / (mean(x) * (PLD4 - PLD1))
      9:   Early-to-late perfusion ratio: (x1 + x2) / (x3 + x4 + eps)
      10:  Relative peak concentration: max(x) / (mean(x) + eps)
    All computed strictly from observed signals and known acquisition parameters.
    """
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]
    x3 = X_raw[:, 2]
    x4 = X_raw[:, 3]

    dt21 = PLDS[1] - PLDS[0]
    dt32 = PLDS[2] - PLDS[1]
    dt43 = PLDS[3] - PLDS[2]
    dt41 = PLDS[3] - PLDS[0]

    # 1-3. Slopes
    s21 = (x2 - x1) / dt21
    s32 = (x3 - x2) / dt32
    s43 = (x4 - x3) / dt43

    # 4-6. Successive decay ratios
    r21 = x2 / np.maximum(x1, eps)
    r32 = x3 / np.maximum(x2, eps)
    r43 = x4 / np.maximum(x3, eps)

    # 7. Apparent decay rate (exponential slope)
    safe_x1 = np.maximum(x1, eps)
    safe_x4 = np.maximum(x4, eps)
    r_app = -np.log(safe_x4 / safe_x1) / dt41

    # 8. Normalized Trapezoidal AUC
    auc = 0.5 * ((x1 + x2) * dt21 + (x2 + x3) * dt32 + (x3 + x4) * dt43)
    mean_x = np.maximum(np.mean(X_raw, axis=1), eps)
    auc_norm = auc / (mean_x * dt41)

    # 9. Early-to-late ratio
    elr = (x1 + x2) / np.maximum(x3 + x4, eps)

    # 10. Relative peak amplitude
    peak_val = np.max(X_raw, axis=1)
    peak_rel = peak_val / mean_x

    feats = np.column_stack([s21, s32, s43, r21, r32, r43, r_app, auc_norm, elr, peak_rel])
    return feats.astype(np.float32)

def fit_feature_scaler(F_tr):
    """Compute mean and std strictly on training set."""
    mu = np.mean(F_tr, axis=0, keepdims=True).astype(np.float32)
    std = np.std(F_tr, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mu, std

def apply_feature_scaler(F, mu, std):
    """Standardize features with frozen training statistics."""
    return ((F - mu) / std).astype(np.float32)

def check_feature_conditioning(F, feature_name):
    """
    Check condition number, variance, and pairwise correlation.
    Returns summary dict.
    """
    cov = np.cov(F, rowvar=False)
    s = np.linalg.svd(cov, compute_uv=False)
    cond_num = float(s[0] / max(s[-1], 1e-12))
    corr = np.corrcoef(F, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    max_corr = float(np.max(np.abs(corr)))

    return {
        "Feature_Set": feature_name,
        "Num_Features": F.shape[1],
        "Condition_Number": cond_num,
        "Max_Pairwise_Correlation": max_corr,
        "Min_Variance": float(np.min(np.var(F, axis=0))),
        "Max_Variance": float(np.max(np.var(F, axis=0)))
    }

