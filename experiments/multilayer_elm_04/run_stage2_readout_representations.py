"""
run_stage2_readout_representations.py
Stage 2: Evaluate Readout Representations on Validation Set for Experiment 4.

Compares:
- Rep A (Control): Penultimate layer only (CBF: h5 [128-d], ATT: h4 [64-d])
- Rep B: Adjacent hidden layers (CBF: [h4, h5] [256-d], ATT: [h3, h4] [128-d])
- Rep C: Early + penultimate (CBF: [h1, h5] [256-d], ATT: [h1, h4] [128-d])
- Rep D: Normalized multi-layer ([h1, h3, h5] [384-d] for CBF; [h1, h2, h4] [192-d] for ATT)
- Rep E: Compact PCA projection of all hidden layers fitted strictly on training data

Evaluates numerical conditioning, stability, and validation set performance across alpha grid.
"""

import os
import sys
import gc
import time
import json
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA

# Environment and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments", "elm_models"))

import memory_config
from physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
)
from train_baseline_A import StandardizedNet, calc_metrics

OUT_DIR = os.path.join(BASE_DIR, "experiments", "multilayer_elm_04")
CACHE_DIR = os.path.join(BASE_DIR, "experiments", "cache")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stage 2] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ---------------------------------------------------------------------------
# 1. Feature Extractor for Multi-Layer Representations
# ---------------------------------------------------------------------------
class MultiLayerExtractor:
    def __init__(self, net, is_cbf=True):
        self.net = net
        self.is_cbf = is_cbf
        self.layers = list(net.backbone.children())
        self.elu_indices = [1, 3, 5, 7, 9] if is_cbf else [1, 3, 5, 7]
        self.pca_model = None

    def fit_pca(self, X_sample_std, n_components):
        """Fit PCA on concatenated all-hidden-layers strictly on training data."""
        with torch.no_grad():
            x_t = torch.from_numpy(X_sample_std.astype(np.float32)).to(device)
            all_feats = self._forward_all_layers(x_t)
            concat_all = torch.cat(list(all_feats.values()), dim=1).detach().cpu().numpy()
            self.pca_model = PCA(n_components=n_components, random_state=42)
            self.pca_model.fit(concat_all)
            self.pca_mean = torch.from_numpy(self.pca_model.mean_.astype(np.float32)).to(device)
            self.pca_W = torch.from_numpy(self.pca_model.components_.T.astype(np.float32)).to(device)
            explained_var = float(np.sum(self.pca_model.explained_variance_ratio_))
            print(f"  [PCA Fit {'CBF' if self.is_cbf else 'ATT'}] {concat_all.shape[1]}-d -> {n_components}-d, explained variance = {explained_var:.4f}")
            return explained_var

    def _forward_all_layers(self, x_t):
        activations = {}
        h = x_t
        h_idx = 1
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx in self.elu_indices:
                activations[f"h{h_idx}"] = h
                h_idx += 1
        return activations

    def get_representation(self, x_t, rep_key):
        with torch.no_grad():
            all_feats = self._forward_all_layers(x_t)
            if self.is_cbf:
                # 5 hidden layers: h1, h2, h3, h4, h5 (128-d each)
                if rep_key == "Rep_A":
                    return all_feats["h5"]
                elif rep_key == "Rep_B":
                    return torch.cat([all_feats["h4"], all_feats["h5"]], dim=1)
                elif rep_key == "Rep_C":
                    return torch.cat([all_feats["h1"], all_feats["h5"]], dim=1)
                elif rep_key == "Rep_D":
                    # Layer-wise L2 normalization
                    h1_norm = all_feats["h1"] / (torch.norm(all_feats["h1"], p=2, dim=1, keepdim=True) + 1e-6)
                    h3_norm = all_feats["h3"] / (torch.norm(all_feats["h3"], p=2, dim=1, keepdim=True) + 1e-6)
                    h5_norm = all_feats["h5"] / (torch.norm(all_feats["h5"], p=2, dim=1, keepdim=True) + 1e-6)
                    return torch.cat([h1_norm, h3_norm, h5_norm], dim=1)
                elif rep_key == "Rep_E":
                    concat_all = torch.cat(list(all_feats.values()), dim=1)
                    return (concat_all - self.pca_mean) @ self.pca_W
                else:
                    raise ValueError(f"Unknown rep_key: {rep_key}")
            else:
                # 4 hidden layers: h1, h2, h3, h4 (64-d each)
                if rep_key == "Rep_A":
                    return all_feats["h4"]
                elif rep_key == "Rep_B":
                    return torch.cat([all_feats["h3"], all_feats["h4"]], dim=1)
                elif rep_key == "Rep_C":
                    return torch.cat([all_feats["h1"], all_feats["h4"]], dim=1)
                elif rep_key == "Rep_D":
                    # Layer-wise L2 normalization
                    h1_norm = all_feats["h1"] / (torch.norm(all_feats["h1"], p=2, dim=1, keepdim=True) + 1e-6)
                    h2_norm = all_feats["h2"] / (torch.norm(all_feats["h2"], p=2, dim=1, keepdim=True) + 1e-6)
                    h4_norm = all_feats["h4"] / (torch.norm(all_feats["h4"], p=2, dim=1, keepdim=True) + 1e-6)
                    return torch.cat([h1_norm, h2_norm, h4_norm], dim=1)
                elif rep_key == "Rep_E":
                    concat_all = torch.cat(list(all_feats.values()), dim=1)
                    return (concat_all - self.pca_mean) @ self.pca_W
                else:
                    raise ValueError(f"Unknown rep_key: {rep_key}")

# ---------------------------------------------------------------------------
# 2. Fast GPU Ridge Accumulator for Feature Readouts
# ---------------------------------------------------------------------------
class FastGPUChunkedELM:
    def __init__(self, n_hidden=512, activation="tanh", seed=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.seed = seed
        self.W = None
        self.b = None
        self.beta = None
        self.input_dim = None

    def _init_weights(self, input_dim):
        self.input_dim = input_dim
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(input_dim)
        w_np = rng.uniform(-scale, scale, size=(input_dim, self.n_hidden)).astype(np.float32)
        b_np = rng.uniform(-scale, scale, size=(1, self.n_hidden)).astype(np.float32)
        self.W = torch.from_numpy(w_np).to(device)
        self.b = torch.from_numpy(b_np).to(device)

    def _activate(self, z):
        if self.activation == "tanh":
            return torch.tanh(z)
        elif self.activation == "elu":
            return torch.nn.functional.elu(z)
        elif self.activation == "relu":
            return torch.relu(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def accumulate_HtH_HtY(self, extractor, rep_key, X_tr_std, Y_tr_std, chunk_size=25_000):
        N = X_tr_std.shape[0]
        dummy_x = torch.zeros((1, 4), device=device)
        sample_rep = extractor.get_representation(dummy_x, rep_key)
        rep_dim = sample_rep.shape[1]

        if self.W is None:
            self._init_weights(rep_dim)

        L = self.n_hidden
        HtH = torch.zeros((L, L), dtype=torch.float64, device=device)
        HtY = torch.zeros((L, 1), dtype=torch.float64, device=device)

        n_chunks = int(np.ceil(N / chunk_size))
        with torch.no_grad():
            for c in range(n_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)

                xc = torch.from_numpy(X_tr_std[start:end].astype(np.float32)).to(device)
                yc = torch.from_numpy(Y_tr_std[start:end].astype(np.float32)).to(device)
                if yc.ndim == 1:
                    yc = yc.unsqueeze(1)

                repc = extractor.get_representation(xc, rep_key)
                hc = self._activate(repc @ self.W + self.b).to(torch.float64)

                HtH += hc.T @ hc
                HtY += hc.T @ yc.to(torch.float64)

                del xc, yc, repc, hc

        return HtH, HtY

    def solve_alpha(self, HtH, HtY, alpha):
        L = self.n_hidden
        I = torch.eye(L, dtype=torch.float64, device=device)
        A = HtH + float(alpha) * I
        beta = torch.linalg.solve(A, HtY).to(torch.float32)
        del A, I
        return beta

    def predict_from_beta(self, extractor, rep_key, X_val_std, beta, chunk_size=25_000):
        N = X_val_std.shape[0]
        preds = np.empty(N, dtype=np.float32)
        n_chunks = int(np.ceil(N / chunk_size))
        with torch.no_grad():
            for c in range(n_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)

                xc = torch.from_numpy(X_val_std[start:end].astype(np.float32)).to(device)
                repc = extractor.get_representation(xc, rep_key)
                hc = self._activate(repc @ self.W + self.b)
                yc = (hc @ beta).squeeze(-1)
                preds[start:end] = yc.cpu().numpy()
                del xc, repc, hc, yc
        return preds

# ---------------------------------------------------------------------------
# 3. Main Stage 2 Sweep Function
# ---------------------------------------------------------------------------
def run_stage2():
    print("=" * 80)
    print("STAGE 2: EVALUATING READOUT REPRESENTATIONS (A, B, C, D, E)")
    print("=" * 80)

    # 1. Load Data and Checkpoints
    norm_path = os.path.join(BASE_DIR, "experiments", "baseline_A", "normalization_stats.npz")
    norm_data = np.load(norm_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    y_cbf_mean = float(norm_data["cbf_mean"])
    y_cbf_std = float(norm_data["cbf_std"])
    y_att_mean = float(norm_data["att_mean"])
    y_att_std = float(norm_data["att_std"])

    cbf_net = StandardizedNet(4, 5, 128, y_cbf_mean, y_cbf_std, CBF_MIN, CBF_MAX).to(device)
    cbf_ckpt_path = os.path.join(BASE_DIR, "experiments", "baseline_A", "CBFnet_G_baselineA.pth")
    cbf_net.load_state_dict(torch.load(cbf_ckpt_path, map_location=device))
    cbf_net.eval()

    att_net = StandardizedNet(4, 4, 64, y_att_mean, y_att_std, ATT_MIN, ATT_MAX).to(device)
    att_ckpt_path = os.path.join(BASE_DIR, "experiments", "baseline_A", "ATTnet_G_baselineA.pth")
    att_net.load_state_dict(torch.load(att_ckpt_path, map_location=device))
    att_net.eval()

    cbf_extractor = MultiLayerExtractor(cbf_net, is_cbf=True)
    att_extractor = MultiLayerExtractor(att_net, is_cbf=False)

    # Load dataset arrays
    X_tr_raw = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    Y_tr_raw = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    X_val_raw = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    Y_val_raw = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    # Standardize inputs using training statistics
    X_tr_std = (X_tr_raw[:] - X_mean) / X_std
    X_val_std = (X_val_raw[:] - X_mean) / X_std

    # Standardize targets
    Y_cbf_tr_std = (Y_tr_raw[:, 0] - y_cbf_mean) / y_cbf_std
    Y_att_tr_std = (Y_tr_raw[:, 1] - y_att_mean) / y_att_std

    Y_cbf_val_phys = Y_val_raw[:, 0].astype(np.float64)
    Y_att_val_phys = Y_val_raw[:, 1].astype(np.float64)
    early_att_val_mask = (Y_att_val_phys >= 0.5) & (Y_att_val_phys < 1.0)

    # Compute Baseline DNN validation metrics
    pred_cbf_dnn = cbf_net.predict_physical(X_val_std)
    pred_att_dnn = att_net.predict_physical(X_val_std)

    m_cbf_dnn = calc_metrics(Y_cbf_val_phys, pred_cbf_dnn)
    m_att_dnn = calc_metrics(Y_att_val_phys, pred_att_dnn)
    m_early_att_dnn = calc_metrics(Y_att_val_phys[early_att_val_mask], pred_att_dnn[early_att_val_mask])

    print("\n--- Baseline DNN Validation Performance ---")
    print(f"  CBF DNN: RMSE={m_cbf_dnn['RMSE']:.4f}, MAE={m_cbf_dnn['MAE']:.4f}, CCC={m_cbf_dnn['CCC']:.4f}, R2={m_cbf_dnn['R2']:.4f}")
    print(f"  ATT DNN: RMSE={m_att_dnn['RMSE']:.4f}, MAE={m_att_dnn['MAE']:.4f}, CCC={m_att_dnn['CCC']:.4f}, R2={m_att_dnn['R2']:.4f}")
    print(f"  Early ATT [0.5, 1.0)s: RMSE={m_early_att_dnn['RMSE']:.4f}, Bias={m_early_att_dnn['Bias']:+.4f}")

    # 2. Fit PCA strictly on Training Subset for Rep E
    print("\nFitting PCA for Representation E strictly on training subset (50,000 samples)...")
    pca_cbf_var = cbf_extractor.fit_pca(X_tr_std[:50_000], n_components=128)
    pca_att_var = att_extractor.fit_pca(X_tr_std[:50_000], n_components=64)

    # 3. Conditioning and Numerical Stability Audit
    print("\nAuditing numerical conditioning of representations on 20,000 training samples...")
    sample_audit = X_tr_std[:20_000]
    rep_keys = ["Rep_A", "Rep_B", "Rep_C", "Rep_D", "Rep_E"]
    audit_rows = []

    for param, extractor in [("CBF", cbf_extractor), ("ATT", att_extractor)]:
        for rk in rep_keys:
            with torch.no_grad():
                x_t = torch.from_numpy(sample_audit.astype(np.float32)).to(device)
                rep = extractor.get_representation(x_t, rk).cpu().numpy()
            
            dim = rep.shape[1]
            mean_norm = float(np.mean(np.linalg.norm(rep, axis=1)))
            cov = np.cov(rep, rowvar=False)
            evals = np.linalg.eigvalsh(cov)
            evals = np.maximum(evals, 1e-12)
            cond_num = float(evals[-1] / evals[0])
            has_nan_inf = bool(np.isnan(rep).any() or np.isinf(rep).any())

            audit_rows.append({
                "Parameter": param,
                "Representation": rk,
                "Dimension": dim,
                "Mean_Sample_Norm": round(mean_norm, 4),
                "Min_Eigenvalue": float(evals[0]),
                "Max_Eigenvalue": float(evals[-1]),
                "Condition_Number": float(cond_num),
                "Has_NaN_or_Inf": has_nan_inf
            })

    audit_df = pd.DataFrame(audit_rows)
    audit_csv = os.path.join(OUT_DIR, "stage2_conditioning_audit.csv")
    audit_df.to_csv(audit_csv, index=False)
    print(f"Conditioning audit saved to {audit_csv}")
    print(audit_df[["Parameter", "Representation", "Dimension", "Mean_Sample_Norm", "Condition_Number"]].to_string())

    # 4. Evaluation of Representations Across Alpha Regularization Grid
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    comparison_results = []

    print("\n" + "=" * 70)
    print("RUNNING STAGE 2 REPRESENTATION EVALUATION (TRAIN: N=1,000,000; VAL: N=100,000)")
    print("=" * 70)

    # -------------------------------------------------------------
    # CBF Representations Sweep
    # -------------------------------------------------------------
    print("\n--- Sweeping CBF Representations ---")
    for rk in rep_keys:
        t0 = time.time()
        elm = FastGPUChunkedELM(n_hidden=512, activation="tanh", seed=42)
        HtH, HtY = elm.accumulate_HtH_HtY(cbf_extractor, rk, X_tr_std, Y_cbf_tr_std, chunk_size=25_000)
        t_acc = time.time() - t0

        best_rmse = float("inf")
        best_alpha = None
        best_metrics = None
        best_pred = None

        for alpha in alphas:
            beta = elm.solve_alpha(HtH, HtY, alpha)
            val_preds_std = elm.predict_from_beta(cbf_extractor, rk, X_val_std, beta, chunk_size=25_000)
            val_preds_phys = np.clip(val_preds_std * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
            m = calc_metrics(Y_cbf_val_phys, val_preds_phys)

            if m["RMSE"] < best_rmse:
                best_rmse = m["RMSE"]
                best_alpha = alpha
                best_metrics = m
                best_pred = val_preds_phys

        del HtH, HtY
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cbf_pct_imp_dnn = (m_cbf_dnn["RMSE"] - best_metrics["RMSE"]) / m_cbf_dnn["RMSE"] * 100.0

        comparison_results.append({
            "Target": "CBF",
            "Representation": rk,
            "Dimension": elm.input_dim,
            "Best_Alpha": best_alpha,
            "Val_RMSE": round(best_metrics["RMSE"], 4),
            "Val_MAE": round(best_metrics["MAE"], 4),
            "Val_CCC": round(best_metrics["CCC"], 4),
            "Val_R2": round(best_metrics["R2"], 4),
            "Val_Bias": round(best_metrics["Bias"], 4),
            "Early_ATT_RMSE": None,
            "Pct_Imp_over_DNN": round(cbf_pct_imp_dnn, 2),
            "Accumulation_Sec": round(t_acc, 2)
        })

        print(f"  [CBF {rk:5s} (dim={elm.input_dim:3d})] Best Alpha={best_alpha:.1e} | Val RMSE={best_metrics['RMSE']:.4f} ({cbf_pct_imp_dnn:+.2f}% vs DNN) | CCC={best_metrics['CCC']:.4f} | Time={t_acc:.1f}s")

    # -------------------------------------------------------------
    # ATT Representations Sweep
    # -------------------------------------------------------------
    print("\n--- Sweeping ATT Representations ---")
    for rk in rep_keys:
        t0 = time.time()
        elm = FastGPUChunkedELM(n_hidden=512, activation="tanh", seed=42)
        HtH, HtY = elm.accumulate_HtH_HtY(att_extractor, rk, X_tr_std, Y_att_tr_std, chunk_size=25_000)
        t_acc = time.time() - t0

        best_rmse = float("inf")
        best_alpha = None
        best_metrics = None
        best_early_rmse = None

        for alpha in alphas:
            beta = elm.solve_alpha(HtH, HtY, alpha)
            val_preds_std = elm.predict_from_beta(att_extractor, rk, X_val_std, beta, chunk_size=25_000)
            val_preds_phys = np.clip(val_preds_std * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
            m = calc_metrics(Y_att_val_phys, val_preds_phys)
            m_early = calc_metrics(Y_att_val_phys[early_att_val_mask], val_preds_phys[early_att_val_mask])

            if m["RMSE"] < best_rmse:
                best_rmse = m["RMSE"]
                best_alpha = alpha
                best_metrics = m
                best_early_rmse = m_early["RMSE"]

        del HtH, HtY
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        att_pct_imp_dnn = (m_att_dnn["RMSE"] - best_metrics["RMSE"]) / m_att_dnn["RMSE"] * 100.0

        comparison_results.append({
            "Target": "ATT",
            "Representation": rk,
            "Dimension": elm.input_dim,
            "Best_Alpha": best_alpha,
            "Val_RMSE": round(best_metrics["RMSE"], 4),
            "Val_MAE": round(best_metrics["MAE"], 4),
            "Val_CCC": round(best_metrics["CCC"], 4),
            "Val_R2": round(best_metrics["R2"], 4),
            "Val_Bias": round(best_metrics["Bias"], 4),
            "Early_ATT_RMSE": round(best_early_rmse, 4),
            "Pct_Imp_over_DNN": round(att_pct_imp_dnn, 2),
            "Accumulation_Sec": round(t_acc, 2)
        })

        print(f"  [ATT {rk:5s} (dim={elm.input_dim:3d})] Best Alpha={best_alpha:.1e} | Val RMSE={best_metrics['RMSE']:.4f} ({att_pct_imp_dnn:+.2f}% vs DNN) | Early RMSE={best_early_rmse:.4f} | CCC={best_metrics['CCC']:.4f} | Time={t_acc:.1f}s")

    # 5. Save Summary Table
    res_df = pd.DataFrame(comparison_results)
    res_csv = os.path.join(OUT_DIR, "stage2_representation_comparison.csv")
    res_df.to_csv(res_csv, index=False)
    print("\n" + "=" * 80)
    print(f"STAGE 2 COMPLETE. Summary saved to {res_csv}")
    print(res_df.to_string())
    print("=" * 80)

if __name__ == "__main__":
    run_stage2()

