"""
run_stage3_nonlinear_readouts.py
Stage 3: Compare Nonlinear Readout Mechanisms on Validation Set for Experiment 4.

Compares:
1. Standard Ridge ELM (Tanh activation)
2. Alternative activation ELM: ELU and ReLU
3. Random Fourier Feature (RFF) Kernel-ELM (Gaussian RBF kernel approximation)
4. Compact Regularized Linear Ridge Readout (direct representation mapping)

Evaluates on the best representations from Stage 2:
- CBF: Rep A (Penultimate 128-d) and Rep C ([h1, h5] 256-d)
- ATT: Rep C ([h1, h4] 128-d) and Rep A (Penultimate 64-d)

Tuned strictly on validation set across hyperparameters and seeds [42, 43, 44, 45, 46].
Outputs stage3_nonlinear_comparison.csv and stage3_seed_variability.csv.
"""

import os
import sys
import gc
import time
import json
import numpy as np
import torch
import pandas as pd

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
print(f"[Stage 3] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ---------------------------------------------------------------------------
# 1. Multi-Layer Feature Extractor
# ---------------------------------------------------------------------------
class MultiLayerExtractor:
    def __init__(self, net, is_cbf=True):
        self.net = net
        self.is_cbf = is_cbf
        self.layers = list(net.backbone.children())
        self.elu_indices = [1, 3, 5, 7, 9] if is_cbf else [1, 3, 5, 7]

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
                if rep_key == "Rep_A":
                    return all_feats["h5"]
                elif rep_key == "Rep_C":
                    return torch.cat([all_feats["h1"], all_feats["h5"]], dim=1)
                else:
                    raise ValueError(f"Unknown rep_key: {rep_key}")
            else:
                if rep_key == "Rep_A":
                    return all_feats["h4"]
                elif rep_key == "Rep_C":
                    return torch.cat([all_feats["h1"], all_feats["h4"]], dim=1)
                else:
                    raise ValueError(f"Unknown rep_key: {rep_key}")

# ---------------------------------------------------------------------------
# 2. General Readout Model Class
# ---------------------------------------------------------------------------
class ReadoutModel:
    def __init__(self, readout_type="elm", activation="tanh", n_hidden=512, rff_gamma=0.1, seed=42):
        self.readout_type = readout_type  # 'elm', 'rff', 'linear'
        self.activation = activation      # 'tanh', 'elu', 'relu'
        self.n_hidden = n_hidden
        self.rff_gamma = float(rff_gamma)
        self.seed = seed
        self.W = None
        self.b = None
        self.input_dim = None

    def _init_weights(self, input_dim):
        self.input_dim = input_dim
        rng = np.random.default_rng(self.seed)
        if self.readout_type == "linear":
            return  # No random hidden layer
        elif self.readout_type == "rff":
            # RFF for Gaussian RBF kernel: exp(-gamma * ||x-x'||^2)
            # W ~ N(0, 2 * gamma), b ~ U[0, 2*pi]
            std_w = np.sqrt(2.0 * self.rff_gamma)
            w_np = rng.normal(0.0, std_w, size=(input_dim, self.n_hidden)).astype(np.float32)
            b_np = rng.uniform(0.0, 2.0 * np.pi, size=(1, self.n_hidden)).astype(np.float32)
            self.W = torch.from_numpy(w_np).to(device)
            self.b = torch.from_numpy(b_np).to(device)
        else:
            # ELM standard initialization
            scale = 1.0 / np.sqrt(input_dim)
            w_np = rng.uniform(-scale, scale, size=(input_dim, self.n_hidden)).astype(np.float32)
            b_np = rng.uniform(-scale, scale, size=(1, self.n_hidden)).astype(np.float32)
            self.W = torch.from_numpy(w_np).to(device)
            self.b = torch.from_numpy(b_np).to(device)

    def _transform(self, rep):
        if self.readout_type == "linear":
            return rep.to(torch.float64)
        elif self.readout_type == "rff":
            # z(x) = sqrt(2/D) * cos(x W + b)
            proj = rep @ self.W + self.b
            return (torch.cos(proj) * np.sqrt(2.0 / self.n_hidden)).to(torch.float64)
        else:
            proj = rep @ self.W + self.b
            if self.activation == "tanh":
                return torch.tanh(proj).to(torch.float64)
            elif self.activation == "elu":
                return torch.nn.functional.elu(proj).to(torch.float64)
            elif self.activation == "relu":
                return torch.relu(proj).to(torch.float64)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")

    def accumulate_HtH_HtY(self, extractor, rep_key, X_tr_std, Y_tr_std, chunk_size=25_000):
        N = X_tr_std.shape[0]
        with torch.no_grad():
            dummy_x = torch.zeros((1, 4), device=device)
            sample_rep = extractor.get_representation(dummy_x, rep_key)
            rep_dim = sample_rep.shape[1]

        if self.W is None and self.readout_type != "linear":
            self._init_weights(rep_dim)
        elif self.readout_type == "linear":
            self.input_dim = rep_dim

        feat_dim = rep_dim if self.readout_type == "linear" else self.n_hidden
        HtH = torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device)
        HtY = torch.zeros((feat_dim, 1), dtype=torch.float64, device=device)

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
                hc = self._transform(repc)

                HtH += hc.T @ hc
                HtY += hc.T @ yc.to(torch.float64)

                del xc, yc, repc, hc

        return HtH, HtY

    def solve_alpha(self, HtH, HtY, alpha):
        feat_dim = HtH.shape[0]
        I = torch.eye(feat_dim, dtype=torch.float64, device=device)
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
                hc = self._transform(repc).to(torch.float32)
                yc = (hc @ beta).squeeze(-1)
                preds[start:end] = yc.cpu().numpy()
                del xc, repc, hc, yc
        return preds

# ---------------------------------------------------------------------------
# 3. Main Stage 3 Execution
# ---------------------------------------------------------------------------
def run_stage3():
    print("=" * 80)
    print("STAGE 3: EVALUATING NONLINEAR READOUT MECHANISMS ON VALIDATION SET")
    print("=" * 80)

    # Load Normalization & Models
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

    # Load Arrays
    X_tr_raw = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    Y_tr_raw = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    X_val_raw = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    Y_val_raw = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    X_tr_std = (X_tr_raw[:] - X_mean) / X_std
    X_val_std = (X_val_raw[:] - X_mean) / X_std

    Y_cbf_tr_std = (Y_tr_raw[:, 0] - y_cbf_mean) / y_cbf_std
    Y_att_tr_std = (Y_tr_raw[:, 1] - y_att_mean) / y_att_std

    Y_cbf_val_phys = Y_val_raw[:, 0].astype(np.float64)
    Y_att_val_phys = Y_val_raw[:, 1].astype(np.float64)
    early_att_val_mask = (Y_att_val_phys >= 0.5) & (Y_att_val_phys < 1.0)

    # Baseline DNN validation metrics
    pred_cbf_dnn = cbf_net.predict_physical(X_val_std)
    pred_att_dnn = att_net.predict_physical(X_val_std)
    m_cbf_dnn = calc_metrics(Y_cbf_val_phys, pred_cbf_dnn)
    m_att_dnn = calc_metrics(Y_att_val_phys, pred_att_dnn)
    m_early_att_dnn = calc_metrics(Y_att_val_phys[early_att_val_mask], pred_att_dnn[early_att_val_mask])

    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    # Candidate Configurations to Evaluate
    readout_configs = [
        # 1. Existing Ridge ELM (Tanh)
        {"name": "ELM_Tanh_512", "type": "elm", "act": "tanh", "hidden": 512, "gamma": 0.0},
        {"name": "ELM_Tanh_1024", "type": "elm", "act": "tanh", "hidden": 1024, "gamma": 0.0},
        # 2. Alternative Activation ELM (ELU and ReLU)
        {"name": "ELM_ELU_512", "type": "elm", "act": "elu", "hidden": 512, "gamma": 0.0},
        {"name": "ELM_ReLU_512", "type": "elm", "act": "relu", "hidden": 512, "gamma": 0.0},
        # 3. Random Fourier Feature (RFF) Kernel-ELM
        {"name": "Kernel_RFF_gamma0.01", "type": "rff", "act": "cos", "hidden": 512, "gamma": 0.01},
        {"name": "Kernel_RFF_gamma0.1", "type": "rff", "act": "cos", "hidden": 512, "gamma": 0.1},
        {"name": "Kernel_RFF_gamma1.0", "type": "rff", "act": "cos", "hidden": 512, "gamma": 1.0},
        # 4. Compact Linear Ridge Readout (Benchmark)
        {"name": "Linear_Ridge", "type": "linear", "act": "identity", "hidden": 0, "gamma": 0.0}
    ]

    results = []

    # -------------------------------------------------------------
    # 1. Sweep CBF Readouts (on Rep A and Rep C)
    # -------------------------------------------------------------
    print("\n--- Sweeping CBF Readout Configurations ---")
    for rk in ["Rep_A", "Rep_C"]:
        for cfg in readout_configs:
            t0 = time.time()
            model = ReadoutModel(
                readout_type=cfg["type"], activation=cfg["act"],
                n_hidden=cfg["hidden"], rff_gamma=cfg["gamma"], seed=42
            )
            HtH, HtY = model.accumulate_HtH_HtY(cbf_extractor, rk, X_tr_std, Y_cbf_tr_std, chunk_size=25_000)
            t_acc = time.time() - t0

            best_rmse = float("inf")
            best_alpha = None
            best_m = None

            for alpha in alphas:
                beta = model.solve_alpha(HtH, HtY, alpha)
                val_preds_std = model.predict_from_beta(cbf_extractor, rk, X_val_std, beta, chunk_size=25_000)
                val_preds_phys = np.clip(val_preds_std * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
                m = calc_metrics(Y_cbf_val_phys, val_preds_phys)

                if m["RMSE"] < best_rmse:
                    best_rmse = m["RMSE"]
                    best_alpha = alpha
                    best_m = m

            del HtH, HtY
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pct_imp = (m_cbf_dnn["RMSE"] - best_m["RMSE"]) / m_cbf_dnn["RMSE"] * 100.0

            results.append({
                "Target": "CBF",
                "Representation": rk,
                "Readout_Config": cfg["name"],
                "Readout_Type": cfg["type"],
                "Activation": cfg["act"],
                "Hidden_Units": cfg["hidden"] if cfg["type"] != "linear" else model.input_dim,
                "Best_Alpha": best_alpha,
                "Val_RMSE": round(best_m["RMSE"], 4),
                "Val_MAE": round(best_m["MAE"], 4),
                "Val_CCC": round(best_m["CCC"], 4),
                "Val_R2": round(best_m["R2"], 4),
                "Val_Bias": round(best_m["Bias"], 4),
                "Early_ATT_RMSE": None,
                "Pct_Imp_over_DNN": round(pct_imp, 2),
                "Training_Sec": round(t_acc, 2)
            })

            print(f"  [CBF {rk} | {cfg['name']:22s}] Best Alpha={best_alpha:.1e} | Val RMSE={best_m['RMSE']:.4f} ({pct_imp:+.2f}%) | Time={t_acc:.1f}s")

    # -------------------------------------------------------------
    # 2. Sweep ATT Readouts (on Rep C and Rep A)
    # -------------------------------------------------------------
    print("\n--- Sweeping ATT Readout Configurations ---")
    for rk in ["Rep_C", "Rep_A"]:
        for cfg in readout_configs:
            t0 = time.time()
            model = ReadoutModel(
                readout_type=cfg["type"], activation=cfg["act"],
                n_hidden=cfg["hidden"], rff_gamma=cfg["gamma"], seed=42
            )
            HtH, HtY = model.accumulate_HtH_HtY(att_extractor, rk, X_tr_std, Y_att_tr_std, chunk_size=25_000)
            t_acc = time.time() - t0

            best_rmse = float("inf")
            best_alpha = None
            best_m = None
            best_early_rmse = None

            for alpha in alphas:
                beta = model.solve_alpha(HtH, HtY, alpha)
                val_preds_std = model.predict_from_beta(att_extractor, rk, X_val_std, beta, chunk_size=25_000)
                val_preds_phys = np.clip(val_preds_std * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
                m = calc_metrics(Y_att_val_phys, val_preds_phys)
                m_early = calc_metrics(Y_att_val_phys[early_att_val_mask], val_preds_phys[early_att_val_mask])

                if m["RMSE"] < best_rmse:
                    best_rmse = m["RMSE"]
                    best_alpha = alpha
                    best_m = m
                    best_early_rmse = m_early["RMSE"]

            del HtH, HtY
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pct_imp = (m_att_dnn["RMSE"] - best_m["RMSE"]) / m_att_dnn["RMSE"] * 100.0

            results.append({
                "Target": "ATT",
                "Representation": rk,
                "Readout_Config": cfg["name"],
                "Readout_Type": cfg["type"],
                "Activation": cfg["act"],
                "Hidden_Units": cfg["hidden"] if cfg["type"] != "linear" else model.input_dim,
                "Best_Alpha": best_alpha,
                "Val_RMSE": round(best_m["RMSE"], 4),
                "Val_MAE": round(best_m["MAE"], 4),
                "Val_CCC": round(best_m["CCC"], 4),
                "Val_R2": round(best_m["R2"], 4),
                "Val_Bias": round(best_m["Bias"], 4),
                "Early_ATT_RMSE": round(best_early_rmse, 4),
                "Pct_Imp_over_DNN": round(pct_imp, 2),
                "Training_Sec": round(t_acc, 2)
            })

            print(f"  [ATT {rk} | {cfg['name']:22s}] Best Alpha={best_alpha:.1e} | Val RMSE={best_m['RMSE']:.4f} ({pct_imp:+.2f}%) | Early={best_early_rmse:.4f} | Time={t_acc:.1f}s")

    # Save Nonlinear Comparison CSV
    res_df = pd.DataFrame(results)
    res_csv = os.path.join(OUT_DIR, "stage3_nonlinear_comparison.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"\nNonlinear comparison table saved to {res_csv}")

    # -------------------------------------------------------------
    # 3. Seed-to-Seed Variability Evaluation across 5 seeds
    # -------------------------------------------------------------
    print("\n--- Evaluating Seed-to-Seed Variability (Seeds: 42, 43, 44, 45, 46) ---")
    seeds = [42, 43, 44, 45, 46]
    seed_records = []

    # Selected candidate models to test across seeds:
    # 1. Historical Baseline: Rep A Tanh 512
    # 2. Rep C Tanh 512
    # 3. Rep A ELU 512
    # 4. Rep A RFF gamma=0.1
    cbf_test_configs = [
        ("Rep_A", "ELM_Tanh_512", "elm", "tanh", 512, 0.0, 1.0),
        ("Rep_C", "ELM_Tanh_512", "elm", "tanh", 512, 0.0, 0.1),
        ("Rep_A", "ELM_ELU_512", "elm", "elu", 512, 0.0, 1.0),
        ("Rep_A", "Kernel_RFF_gamma0.1", "rff", "cos", 512, 0.1, 1.0)
    ]

    att_test_configs = [
        ("Rep_A", "ELM_Tanh_512", "elm", "tanh", 512, 0.0, 10.0),
        ("Rep_C", "ELM_Tanh_512", "elm", "tanh", 512, 0.0, 10.0),
        ("Rep_A", "ELM_ELU_512", "elm", "elu", 512, 0.0, 10.0),
        ("Rep_A", "Kernel_RFF_gamma0.1", "rff", "cos", 512, 0.1, 10.0)
    ]

    for seed in seeds:
        print(f"\nSeed {seed}:")
        # CBF
        for rk, name, rtype, act, nh, gamma, alpha in cbf_test_configs:
            model = ReadoutModel(readout_type=rtype, activation=act, n_hidden=nh, rff_gamma=gamma, seed=seed)
            HtH, HtY = model.accumulate_HtH_HtY(cbf_extractor, rk, X_tr_std, Y_cbf_tr_std, chunk_size=25_000)
            beta = model.solve_alpha(HtH, HtY, alpha)
            val_preds_std = model.predict_from_beta(cbf_extractor, rk, X_val_std, beta, chunk_size=25_000)
            val_preds_phys = np.clip(val_preds_std * y_cbf_std + y_cbf_mean, CBF_MIN, CBF_MAX)
            m = calc_metrics(Y_cbf_val_phys, val_preds_phys)
            del HtH, HtY, beta
            seed_records.append({
                "Seed": seed,
                "Target": "CBF",
                "Representation": rk,
                "Model_Name": name,
                "Val_RMSE": round(m["RMSE"], 4),
                "Val_MAE": round(m["MAE"], 4),
                "Val_CCC": round(m["CCC"], 4),
                "Val_R2": round(m["R2"], 4),
                "Early_ATT_RMSE": None
            })

        # ATT
        for rk, name, rtype, act, nh, gamma, alpha in att_test_configs:
            model = ReadoutModel(readout_type=rtype, activation=act, n_hidden=nh, rff_gamma=gamma, seed=seed)
            HtH, HtY = model.accumulate_HtH_HtY(att_extractor, rk, X_tr_std, Y_att_tr_std, chunk_size=25_000)
            beta = model.solve_alpha(HtH, HtY, alpha)
            val_preds_std = model.predict_from_beta(att_extractor, rk, X_val_std, beta, chunk_size=25_000)
            val_preds_phys = np.clip(val_preds_std * y_att_std + y_att_mean, ATT_MIN, ATT_MAX)
            m = calc_metrics(Y_att_val_phys, val_preds_phys)
            m_early = calc_metrics(Y_att_val_phys[early_att_val_mask], val_preds_phys[early_att_val_mask])
            del HtH, HtY, beta
            seed_records.append({
                "Seed": seed,
                "Target": "ATT",
                "Representation": rk,
                "Model_Name": name,
                "Val_RMSE": round(m["RMSE"], 4),
                "Val_MAE": round(m["MAE"], 4),
                "Val_CCC": round(m["CCC"], 4),
                "Val_R2": round(m["R2"], 4),
                "Early_ATT_RMSE": round(m_early["RMSE"], 4)
            })

    seed_df = pd.DataFrame(seed_records)
    seed_csv = os.path.join(OUT_DIR, "stage3_seed_variability.csv")
    seed_df.to_csv(seed_csv, index=False)
    print(f"\nSeed variability results saved to {seed_csv}")

    # Summary table across seeds
    summary_seed = seed_df.groupby(["Target", "Representation", "Model_Name"]).agg(
        Mean_RMSE=("Val_RMSE", "mean"),
        Std_RMSE=("Val_RMSE", "std"),
        Mean_CCC=("Val_CCC", "mean"),
        Std_CCC=("Val_CCC", "std")
    ).reset_index()
    print("\n--- Seed Variability Summary (Mean +/- Std) ---")
    print(summary_seed.to_string())

if __name__ == "__main__":
    run_stage3()

