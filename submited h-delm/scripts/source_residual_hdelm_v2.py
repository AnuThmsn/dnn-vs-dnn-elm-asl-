"""
residual_hdelm_v2.py
Proposed Residual Hybrid DNN-ELM (H-DELM V2) implementation:
1. Freezes the 4x64 DNN-G backbone and extracts 64-dim penultimate activations.
2. Extracts exactly 15 signal-derived physics features:
   - 4 original PLD measurements (1.525, 2.025, 2.525, 3.025 s)
   - 3 inter-PLD slopes: (x_{i+1} - x_i) / (w_{i+1} - w_i)
   - 3 consecutive signal ratios: x_{i+1} / max(x_i, eps)
   - Signal mean
   - Signal variance
   - Signal minimum
   - Signal maximum
   - Trapezoidal area under the signal curve (AUC)
3. Combines DNN features + 15 physics features -> 79-dimensional feature vector.
4. Standardizes the 79-dim representation with mean and std fitted exclusively on calibration data.
5. Fits Extreme Learning Machine (ELM) with 512 random hidden neurons, tanh activation,
   and analytical ridge regression on the residual target:
       r = Y_true - Y_DNN
6. Full save/load artifact functionality to enable standalone inference without retraining.
7. Hyperparameter sensitivity analysis on lambda grid using calibration-train / calibration-val.
"""

import os
import json
import numpy as np
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLDS_DEFAULT = np.array([1.525, 2.025, 2.525, 3.025], dtype=np.float64)

PHYSICS_FEATURE_NAMES = [
    "pld1_signal", "pld2_signal", "pld3_signal", "pld4_signal",
    "slope_pld2_pld1", "slope_pld3_pld2", "slope_pld4_pld3",
    "ratio_pld2_pld1", "ratio_pld3_pld2", "ratio_pld4_pld3",
    "signal_mean", "signal_var", "signal_min", "signal_max",
    "trapezoidal_auc"
]

def extract_15_physics_features(X_raw, plds=PLDS_DEFAULT, eps=1e-3):
    """
    Extracts exactly 15 signal-derived physics features from observed 4-PLD signals.
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

    # Slopes
    s21 = (x2 - x1) / dw21
    s32 = (x3 - x2) / dw32
    s43 = (x4 - x3) / dw43

    # Ratios
    r21 = x2 / np.maximum(x1, eps)
    r32 = x3 / np.maximum(x2, eps)
    r43 = x4 / np.maximum(x3, eps)

    # Moments & bounds
    sig_mean = np.mean(X_raw, axis=1)
    sig_var = np.var(X_raw, axis=1)
    sig_min = np.min(X_raw, axis=1)
    sig_max = np.max(X_raw, axis=1)

    # Trapezoidal AUC
    auc = 0.5 * ((x1 + x2) * dw21 + (x2 + x3) * dw32 + (x3 + x4) * dw43)

    features = np.column_stack([
        x1, x2, x3, x4,
        s21, s32, s43,
        r21, r32, r43,
        sig_mean, sig_var, sig_min, sig_max,
        auc
    ])
    return features.astype(np.float32)

class ResidualHDELMV2:
    def __init__(
        self,
        dnn_net=None,
        y_min=0.0,
        y_max=100.0,
        n_hidden=512,
        ridge_lambda=1e-2,
        seed=42,
        plds=PLDS_DEFAULT
    ):
        self.dnn_net = dnn_net
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.n_hidden = int(n_hidden)
        self.ridge_lambda = float(ridge_lambda)
        self.seed = int(seed)
        self.plds = np.asarray(plds, dtype=np.float64)

        # Scaler for 79-dimensional features
        self.feat_mean = None  # (1, 79)
        self.feat_std = None   # (1, 79)

        # ELM weights
        self.W = None          # (79, 512)
        self.b = None          # (1, 512)
        self.beta = None       # (512, 1)

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
        if self.dnn_net is None:
            raise ValueError("dnn_net must be loaded before extracting features.")
        f_dnn = self.dnn_net.penultimate_features(X_n)
        f_phys = extract_15_physics_features(X_raw, plds=self.plds)
        f_comb = np.hstack([f_dnn, f_phys])
        return f_comb

    def grid_search_lambda(
        self,
        X_raw_tr,
        X_n_tr,
        Y_tr_phys,
        X_raw_val,
        X_n_val,
        Y_val_phys,
        lambda_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        chunk_size=50000
    ):
        """
        Performs sensitivity analysis over ridge lambda on calibration-train and calibration-val subsets.
        Never touches the independent test cohort.
        """
        print(f"--- Hyperparameter Sensitivity Analysis: Lambda Grid {list(lambda_grid)} ---")
        # 1. Feature extraction on calibration-train
        F_tr = self.extract_combined_features(X_raw_tr, X_n_tr)
        feat_mean = np.mean(F_tr, axis=0, keepdims=True).astype(np.float32)
        feat_std = np.std(F_tr, axis=0, keepdims=True).astype(np.float32)
        feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

        Z_tr = ((F_tr - feat_mean) / feat_std).astype(np.float32)
        del F_tr

        # 2. Features on calibration-val
        F_val = self.extract_combined_features(X_raw_val, X_n_val)
        Z_val = ((F_val - feat_mean) / feat_std).astype(np.float32)
        del F_val

        # 3. Residual targets on train and validation
        y_dnn_tr = self.dnn_net.predict_physical(X_n_tr)
        r_tr = (Y_tr_phys - y_dnn_tr).astype(np.float32)

        y_dnn_val = self.dnn_net.predict_physical(X_n_val)
        r_val = (Y_val_phys - y_dnn_val).astype(np.float32)

        # 4. Hidden activations
        self._init_elm_weights(input_dim=79)
        L = self.n_hidden
        HtH = torch.zeros((L, L), dtype=torch.float64, device=device)
        HtR = torch.zeros((L, 1), dtype=torch.float64, device=device)

        N_tr = Z_tr.shape[0]
        n_chunks = int(np.ceil(N_tr / chunk_size))
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N_tr)
            zc = torch.from_numpy(Z_tr[start:end]).to(device)
            rc = torch.from_numpy(r_tr[start:end]).unsqueeze(1).to(device)
            hc = torch.tanh(zc @ self.W + self.b).to(torch.float64)
            HtH += hc.T @ hc
            HtR += hc.T @ rc.to(torch.float64)
            del zc, rc, hc

        # Compute validation hidden representation
        zc_val = torch.from_numpy(Z_val).to(device)
        hc_val = torch.tanh(zc_val @ self.W + self.b).to(torch.float32)

        results = []
        I = torch.eye(L, dtype=torch.float64, device=device)
        for lam in lambda_grid:
            A = HtH + float(lam) * I
            beta_lam = torch.linalg.solve(A, HtR).to(torch.float32)

            # Predict on val
            r_pred_val = (hc_val @ beta_lam).squeeze(-1).cpu().numpy()
            y_pred_val = np.clip(y_dnn_val + r_pred_val, self.y_min, self.y_max)

            # Metrics
            err = y_pred_val - Y_val_phys
            val_mse = float(np.mean(err ** 2))
            val_rmse = float(np.sqrt(val_mse))
            val_mae = float(np.mean(np.abs(err)))
            val_bias = float(np.mean(err))

            results.append({
                "ridge_lambda": float(lam),
                "val_mse": val_mse,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_bias": val_bias
            })
            print(f"  lambda={lam:<7.0e} | Val RMSE={val_rmse:.4f} | Val MAE={val_mae:.4f} | Val Bias={val_bias:+.4f}")

        del zc_val, hc_val, HtH, HtR, I
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results_df = pd.DataFrame(results)
        best_idx = results_df["val_rmse"].idxmin()
        best_lambda = results_df.loc[best_idx, "ridge_lambda"]
        print(f"Optimal lambda selected by calibration-validation: {best_lambda} (RMSE={results_df.loc[best_idx, 'val_rmse']:.4f})")
        return results_df, best_lambda

    def fit(self, X_raw_cal, X_cal_n, Y_cal_phys, chunk_size=50000):
        """
        Fits Residual H-DELM exclusively on calibration dataset.
        """
        N = len(X_raw_cal)
        print(f"Extracting 79-dimensional representation on calibration set (N={N:,})...")
        F_cal = self.extract_combined_features(X_raw_cal, X_cal_n)
        assert F_cal.shape[1] == 79, f"Expected 79 features, got {F_cal.shape[1]}"

        # Standardization statistics fit strictly on calibration data
        self.feat_mean = np.mean(F_cal, axis=0, keepdims=True).astype(np.float32)
        self.feat_std = np.std(F_cal, axis=0, keepdims=True).astype(np.float32)
        self.feat_std = np.where(self.feat_std < 1e-6, 1.0, self.feat_std)

        Z_cal = ((F_cal - self.feat_mean) / self.feat_std).astype(np.float32)
        del F_cal

        # Compute DNN predictions and residual targets
        print("Computing frozen DNN baseline predictions and target residuals on calibration set...")
        y_dnn_cal = self.dnn_net.predict_physical(X_cal_n)
        r_cal = (Y_cal_phys - y_dnn_cal).astype(np.float32)

        # Initialize ELM weights
        self._init_elm_weights(input_dim=79)

        # Solve analytical ridge regression in float64
        print(f"Solving analytical ridge regression (H={self.n_hidden}, lambda={self.ridge_lambda})...")
        L = self.n_hidden
        HtH = torch.zeros((L, L), dtype=torch.float64, device=device)
        HtR = torch.zeros((L, 1), dtype=torch.float64, device=device)

        n_chunks = int(np.ceil(N / chunk_size))
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)

            zc = torch.from_numpy(Z_cal[start:end]).to(device)
            rc = torch.from_numpy(r_cal[start:end]).unsqueeze(1).to(device)

            hc = torch.tanh(zc @ self.W + self.b).to(torch.float64)
            HtH += hc.T @ hc
            HtR += hc.T @ rc.to(torch.float64)
            del zc, rc, hc

        I = torch.eye(L, dtype=torch.float64, device=device)
        A = HtH + self.ridge_lambda * I
        self.beta = torch.linalg.solve(A, HtR).to(torch.float32)

        del HtH, HtR, A, I
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Residual H-DELM V2 fit complete (beta norm: {torch.norm(self.beta).item():.4f}).")
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

    def save_artifact(self, filepath):
        """
        Saves complete ELM artifact:
        - Projection weights W
        - Projection biases b
        - Ridge output weights beta
        - Feature mean and standard deviation
        - Hidden neuron count, ridge lambda, seed
        - Physiological output bounds
        - Feature extraction configuration
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        w_np = self.W.cpu().numpy() if isinstance(self.W, torch.Tensor) else self.W
        b_np = self.b.cpu().numpy() if isinstance(self.b, torch.Tensor) else self.b
        beta_np = self.beta.cpu().numpy() if isinstance(self.beta, torch.Tensor) else self.beta

        metadata = {
            "n_hidden": self.n_hidden,
            "ridge_lambda": self.ridge_lambda,
            "seed": self.seed,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "input_dim": 79,
            "dnn_penultimate_dim": 64,
            "physics_features_dim": 15,
            "physics_feature_names": PHYSICS_FEATURE_NAMES,
            "plds": self.plds.tolist(),
            "activation": "tanh"
        }

        np.savez_compressed(
            filepath,
            W=w_np,
            b=b_np,
            beta=beta_np,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
            metadata=json.dumps(metadata)
        )
        print(f"Saved complete H-DELM artifact to {filepath} ({os.path.getsize(filepath):,} bytes)")

    @classmethod
    def load_artifact(cls, filepath, dnn_net=None):
        """
        Loads H-DELM from saved artifact.
        """
        data = np.load(filepath)
        metadata = json.loads(str(data["metadata"]))

        instance = cls(
            dnn_net=dnn_net,
            y_min=metadata["y_min"],
            y_max=metadata["y_max"],
            n_hidden=metadata["n_hidden"],
            ridge_lambda=metadata["ridge_lambda"],
            seed=metadata["seed"],
            plds=metadata["plds"]
        )

        w_np = data["W"]
        b_np = data["b"]
        beta_np = data["beta"]
        instance.feat_mean = data["feat_mean"]
        instance.feat_std = data["feat_std"]

        instance.W = torch.from_numpy(w_np).to(device)
        instance.b = torch.from_numpy(b_np).to(device)
        instance.beta = torch.from_numpy(beta_np).to(device)

        print(f"Loaded H-DELM artifact from {filepath} (H={instance.n_hidden}, lambda={instance.ridge_lambda})")
        return instance

