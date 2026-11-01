"""
residual_hdelm.py
Proposed Residual Hybrid DNN-ELM (H-DELM) implementation:
1. Freezes the 4x64 DNN-G backbone and extracts 64-dim penultimate activations.
2. Extracts exactly 15 signal-derived physics features:
   - 4 original PLD measurements
   - 3 inter-PLD slopes
   - 3 consecutive signal ratios
   - Signal mean
   - Signal variance
   - Signal minimum
   - Signal maximum
   - Trapezoidal area under the signal curve
3. Combines DNN features + 15 physics features -> 79-dimensional feature vector.
4. Standardizes the 79-dim representation with mean and std fitted on training set.
5. Fits Extreme Learning Machine (ELM) with 512 random hidden neurons, tanh activation,
   and analytical ridge regression on the residual target:
       r = Y_true - Y_DNN
6. Final inference:
       Y_HDELM = clip(Y_DNN + predicted_residual, [y_min, y_max])
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLDS = np.array([1.525, 2.025, 2.525, 3.025], dtype=np.float64)

def extract_15_physics_features(X_raw, plds=PLDS, eps=1e-3):
    """
    Extracts exactly 15 signal-derived physics features from observed 4-PLD signals:
      1-4:   4 original PLD measurements: x1, x2, x3, x4
      5-7:   3 inter-PLD slopes: (x_{i+1} - x_i) / (w_{i+1} - w_i)
      8-10:  3 consecutive signal ratios: x_{i+1} / (x_i + eps)
      11:    Signal mean across PLDs
      12:    Signal variance across PLDs
      13:    Signal minimum across PLDs
      14:    Signal maximum across PLDs
      15:    Trapezoidal area under the signal curve (AUC)
    """
    X_raw = np.asarray(X_raw, dtype=np.float64)
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]
    x3 = X_raw[:, 2]
    x4 = X_raw[:, 3]

    w1, w2, w3, w4 = plds[0], plds[1], plds[2], plds[3]
    dw21 = w2 - w1
    dw32 = w3 - w2
    dw43 = w4 - w3

    # 1-4: Raw PLD signals
    # 5-7: Inter-PLD slopes
    s21 = (x2 - x1) / dw21
    s32 = (x3 - x2) / dw32
    s43 = (x4 - x3) / dw43

    # 8-10: Consecutive ratios
    r21 = x2 / np.maximum(x1, eps)
    r32 = x3 / np.maximum(x2, eps)
    r43 = x4 / np.maximum(x3, eps)

    # 11: Signal mean
    sig_mean = np.mean(X_raw, axis=1)

    # 12: Signal variance
    sig_var = np.var(X_raw, axis=1)

    # 13: Signal minimum
    sig_min = np.min(X_raw, axis=1)

    # 14: Signal maximum
    sig_max = np.max(X_raw, axis=1)

    # 15: Trapezoidal AUC
    auc = 0.5 * ((x1 + x2) * dw21 + (x2 + x3) * dw32 + (x3 + x4) * dw43)

    features = np.column_stack([
        x1, x2, x3, x4,
        s21, s32, s43,
        r21, r32, r43,
        sig_mean, sig_var, sig_min, sig_max,
        auc
    ])
    return features.astype(np.float32)

class ResidualHDELM:
    def __init__(self, dnn_net, y_min, y_max, n_hidden=512, ridge_lambda=1e-2, seed=42):
        self.dnn_net = dnn_net
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.n_hidden = n_hidden
        self.ridge_lambda = float(ridge_lambda)
        self.seed = seed

        # Scaler for 79-dimensional features
        self.feat_mean = None
        self.feat_std = None

        # ELM weights
        self.W = None      # (79, 512)
        self.b = None      # (1, 512)
        self.beta = None   # (512, 1)

    def _init_elm_weights(self, input_dim=79):
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(input_dim)
        w_np = rng.uniform(-scale, scale, size=(input_dim, self.n_hidden)).astype(np.float32)
        b_np = rng.uniform(-scale, scale, size=(1, self.n_hidden)).astype(np.float32)
        self.W = torch.from_numpy(w_np).to(device)
        self.b = torch.from_numpy(b_np).to(device)

    def extract_combined_features(self, X_raw, X_n):
        """
        Combines 64-dim DNN penultimate activations + 15 physics features -> 79-dim representation.
        """
        # 1. 64-dim DNN features
        f_dnn = self.dnn_net.penultimate_features(X_n)
        # 2. 15 physics features
        f_phys = extract_15_physics_features(X_raw)
        # 3. Concatenate
        f_comb = np.hstack([f_dnn, f_phys])
        return f_comb

    def fit(self, X_raw_tr, X_tr_n, Y_tr_phys, chunk_size=50000):
        """
        Fits ELM on residuals r = Y_tr_phys - Y_DNN_tr.
        """
        print(f"Extracting 79-dimensional representation for training set (N={len(X_raw_tr):,})...")
        F_tr = self.extract_combined_features(X_raw_tr, X_tr_n)
        assert F_tr.shape[1] == 79, f"Expected 79 features, got {F_tr.shape[1]}"

        # Fit feature standardization statistics on train only
        self.feat_mean = np.mean(F_tr, axis=0, keepdims=True).astype(np.float32)
        self.feat_std = np.std(F_tr, axis=0, keepdims=True).astype(np.float32)
        self.feat_std = np.where(self.feat_std < 1e-6, 1.0, self.feat_std)

        Z_tr = ((F_tr - self.feat_mean) / self.feat_std).astype(np.float32)
        del F_tr

        # Compute DNN predictions and residuals
        print("Computing frozen DNN baseline predictions and target residuals...")
        y_dnn_tr = self.dnn_net.predict_physical(X_tr_n)
        r_tr = (Y_tr_phys - y_dnn_tr).astype(np.float32)

        # Initialize ELM weights
        self._init_elm_weights(input_dim=79)

        # Accumulate H^T H and H^T r in chunks on GPU (or CPU) in float64 precision
        print(f"Solving analytical ridge regression (H={self.n_hidden}, lambda={self.ridge_lambda})...")
        N = Z_tr.shape[0]
        L = self.n_hidden
        HtH = torch.zeros((L, L), dtype=torch.float64, device=device)
        HtR = torch.zeros((L, 1), dtype=torch.float64, device=device)

        n_chunks = int(np.ceil(N / chunk_size))
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)

            zc = torch.from_numpy(Z_tr[start:end]).to(device)
            rc = torch.from_numpy(r_tr[start:end]).unsqueeze(1).to(device)

            # Hidden activations with tanh
            hc = torch.tanh(zc @ self.W + self.b).to(torch.float64)
            HtH += hc.T @ hc
            HtR += hc.T @ rc.to(torch.float64)

            del zc, rc, hc

        # Regularized solve: beta = (HtH + lambda * I)^(-1) HtR
        I = torch.eye(L, dtype=torch.float64, device=device)
        A = HtH + self.ridge_lambda * I
        self.beta = torch.linalg.solve(A, HtR).to(torch.float32)

        del HtH, HtR, A, I
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Residual H-DELM fitting completed successfully.")
        return self

    def predict_residual(self, X_raw, X_n, chunk_size=50000):
        """Predicts residual correction r_hat."""
        F = self.extract_combined_features(X_raw, X_n)
        Z = ((F - self.feat_mean) / self.feat_std).astype(np.float32)
        del F

        N = Z.shape[0]
        r_preds = np.empty(N, dtype=np.float32)
        n_chunks = int(np.ceil(N / chunk_size))

        with torch.no_grad():
            for c in range(n_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                zc = torch.from_numpy(Z[start:end]).to(device)
                hc = torch.tanh(zc @ self.W + self.b)
                rc = (hc @ self.beta).squeeze(-1)
                r_preds[start:end] = rc.cpu().numpy()
                del zc, hc, rc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return r_preds

    def predict_physical(self, X_raw, X_n):
        """
        Final prediction:
            Y_HDELM = clip(Y_DNN + predicted_residual, [y_min, y_max])
        """
        y_dnn = self.dnn_net.predict_physical(X_n)
        r_hat = self.predict_residual(X_raw, X_n)
        y_hdelm = np.clip(y_dnn + r_hat, self.y_min, self.y_max)
        return y_hdelm, y_dnn, r_hat

