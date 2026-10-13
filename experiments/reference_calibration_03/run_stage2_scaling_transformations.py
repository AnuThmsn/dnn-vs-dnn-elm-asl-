"""
run_stage2_scaling_transformations.py
Stage 2: Evaluate numerical scaling, latent standardization, and target transformations
for CBF and ATT estimation under fixed 4-PLD protocol.
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
OUT_DIR = os.path.join(EXP_DIR, "reference_calibration_03")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.append(EXP_DIR)
sys.path.append(os.path.join(EXP_DIR, "elm_models"))

import physics
from train_baseline_A import StandardizedNet, calc_metrics
from elm_engine import GPUChunkedELM, device

class FastGPUChunkedELM(GPUChunkedELM):
    """ELM supporting single H^T H accumulation and instant alpha regularizations."""
    def compute_HtH_HtY(self, X_array, y_array, chunk_size=25_000):
        N, d = X_array.shape
        if self.W is None:
            self._init_weights(d)

        L = self.n_hidden
        HtH = torch.zeros((L, L), dtype=torch.float64, device=device)
        HtY = torch.zeros((L, 1), dtype=torch.float64, device=device)

        n_chunks = int(np.ceil(N / chunk_size))
        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, N)

            xc = torch.from_numpy(X_array[start:end].astype(np.float32)).to(device)
            yc = torch.from_numpy(y_array[start:end].astype(np.float32)).to(device)
            if yc.ndim == 1:
                yc = yc.unsqueeze(1)

            hc = self._activate(xc @ self.W + self.b).to(torch.float64)
            HtH += hc.T @ hc
            HtY += hc.T @ yc.to(torch.float64)
            del xc, yc, hc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return HtH, HtY

    def solve_alpha(self, HtH, HtY, alpha):
        self.alpha = float(alpha)
        L = self.n_hidden
        I = torch.eye(L, dtype=torch.float64, device=device)
        A = HtH + self.alpha * I
        self.beta = torch.linalg.solve(A, HtY).to(torch.float32)
        del A, I
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, X_array, chunk_size=25_000):
        N = X_array.shape[0]
        preds = np.empty(N, dtype=np.float32)
        n_chunks = int(np.ceil(N / chunk_size))
        with torch.no_grad():
            for c in range(n_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)

                xc = torch.from_numpy(X_array[start:end].astype(np.float32)).to(device)
                hc = self._activate(xc @ self.W + self.b)
                yc = (hc @ self.beta).squeeze(-1)
                preds[start:end] = yc.cpu().numpy()
                del xc, hc, yc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return preds


def main():
    print("=" * 80)
    print("EXPERIMENT 3 - STAGE 2: NUMERICAL SCALING & TARGET TRANSFORMATIONS")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Normalization & Baseline Models
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    att_net = StandardizedNet(4, 4, 64, att_mean, att_std, physics.ATT_MIN, physics.ATT_MAX).to(device)
    att_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "ATTnet_G_baselineA.pth"), map_location=device, weights_only=False))
    att_net.eval()

    cbf_net = StandardizedNet(4, 5, 128, cbf_mean, cbf_std, physics.CBF_MIN, physics.CBF_MAX).to(device)
    cbf_net.load_state_dict(torch.load(os.path.join(BASE_DIR, "CBFnet_G_baselineA.pth"), map_location=device, weights_only=False))
    cbf_net.eval()

    # 2. Datasets
    # 100k training sample (seed 2026) for fitting transformations & ELM
    N_tr = 100_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    X_train_raw = np.array(x_tr_mmap[idx_tr], dtype=np.float32)
    Y_train_raw = np.array(y_tr_mmap[idx_tr], dtype=np.float32)

    N_val = len(x_val_mmap)
    X_val_raw = np.array(x_val_mmap[:], dtype=np.float32)
    Y_val_raw = np.array(y_val_mmap[:], dtype=np.float32)

    # Standardized inputs for DNN
    X_train_dnn = (X_train_raw - X_mean) / X_std
    X_val_dnn = (X_val_raw - X_mean) / X_std

    # Baseline DNN predictions on Val
    print("\nEvaluating Baseline DNN on Validation...")
    cbf_dnn_val = cbf_net.predict_physical(X_val_dnn)
    att_dnn_val = att_net.predict_physical(X_val_dnn)

    cbf_dnn_metrics = calc_metrics(Y_val_raw[:, 0], cbf_dnn_val)
    att_dnn_metrics = calc_metrics(Y_val_raw[:, 1], att_dnn_val)
    print(f"Baseline DNN Val CBF RMSE: {cbf_dnn_metrics['RMSE']:.4f}")
    print(f"Baseline DNN Val ATT RMSE: {att_dnn_metrics['RMSE']:.4f}")

    # Extract Penultimate Latents in chunks
    print("\nExtracting Penultimate Latents...")
    chunk_sz = 10_000
    F_tr_cbf = []
    F_val_cbf = []
    for i in range(0, N_tr, chunk_sz):
        F_tr_cbf.append(cbf_net.penultimate_features(X_train_dnn[i:i+chunk_sz]))
    for i in range(0, N_val, chunk_sz):
        F_val_cbf.append(cbf_net.penultimate_features(X_val_dnn[i:i+chunk_sz]))
    F_tr_cbf = np.vstack(F_tr_cbf)
    F_val_cbf = np.vstack(F_val_cbf)

    F_tr_att = []
    F_val_att = []
    for i in range(0, N_tr, chunk_sz):
        F_tr_att.append(att_net.penultimate_features(X_train_dnn[i:i+chunk_sz]))
    for i in range(0, N_val, chunk_sz):
        F_val_att.append(att_net.penultimate_features(X_val_dnn[i:i+chunk_sz]))
    F_tr_att = np.vstack(F_tr_att)
    F_val_att = np.vstack(F_val_att)
    print(f"CBF Latents: {F_tr_cbf.shape}, ATT Latents: {F_tr_att.shape}")

    del cbf_net, att_net
    torch.cuda.empty_cache()

    # =========================================================================
    # PART 1: TARGET TRANSFORMATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: TARGET TRANSFORMATIONS FOR FEATURE ELM (H=2048, alpha sweep)")
    print("=" * 80)

    target_results = []
    seeds = [42, 101, 2024]
    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

    # --- CBF Target Transformations ---
    cbf_transforms = {
        "Standardized": {
            "forward": lambda y: (y - cbf_mean) / cbf_std,
            "inverse": lambda y_hat: np.clip(y_hat * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
        },
        "MinMax_[-1,1]": {
            "forward": lambda y: 2.0 * (y - physics.CBF_MIN) / (physics.CBF_MAX - physics.CBF_MIN) - 1.0,
            "inverse": lambda y_hat: np.clip(physics.CBF_MIN + 0.5 * (y_hat + 1.0) * (physics.CBF_MAX - physics.CBF_MIN), physics.CBF_MIN, physics.CBF_MAX)
        },
        "Raw_Physical": {
            "forward": lambda y: y,
            "inverse": lambda y_hat: np.clip(y_hat, physics.CBF_MIN, physics.CBF_MAX)
        }
    }

    for t_name, t_funcs in cbf_transforms.items():
        print(f"\nEvaluating CBF Target Transform: {t_name}")
        y_tr_t = t_funcs["forward"](Y_train_raw[:, 0])

        # Precompute HtH and HtY for each seed
        seed_cache = []
        for seed in seeds:
            elm = FastGPUChunkedELM(n_hidden=2048, activation="tanh", seed=seed)
            HtH, HtY = elm.compute_HtH_HtY(F_tr_cbf, y_tr_t)
            seed_cache.append((elm, HtH, HtY))

        best_alpha = None
        best_rmse = float("inf")

        for alpha in alphas:
            rmse_seeds = []
            for elm, HtH, HtY in seed_cache:
                elm.solve_alpha(HtH, HtY, alpha)
                pred_raw = elm.predict(F_val_cbf)
                pred_phys = t_funcs["inverse"](pred_raw)
                m = calc_metrics(Y_val_raw[:, 0], pred_phys)
                rmse_seeds.append(m["RMSE"])
            avg_rmse = float(np.mean(rmse_seeds))
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_alpha = alpha

        # Free seed cache
        del seed_cache
        torch.cuda.empty_cache()

        # Re-evaluate with 5 seeds at best alpha
        preds_5 = []
        for seed in [42, 101, 2024, 7, 99]:
            elm = FastGPUChunkedELM(n_hidden=2048, activation="tanh", seed=seed)
            HtH, HtY = elm.compute_HtH_HtY(F_tr_cbf, y_tr_t)
            elm.solve_alpha(HtH, HtY, best_alpha)
            preds_5.append(t_funcs["inverse"](elm.predict(F_val_cbf)))
            del HtH, HtY, elm
            torch.cuda.empty_cache()

        avg_pred = np.mean(preds_5, axis=0)
        final_m = calc_metrics(Y_val_raw[:, 0], avg_pred)
        gain_vs_dnn = (cbf_dnn_metrics["RMSE"] - final_m["RMSE"]) / cbf_dnn_metrics["RMSE"] * 100.0
        print(f"  Best alpha={best_alpha}: Val RMSE={final_m['RMSE']:.4f}, MAE={final_m['MAE']:.4f}, Bias={final_m['Bias']:.4f}, Gain={gain_vs_dnn:+.2f}%")

        target_results.append({
            "Target": "CBF",
            "Transformation": t_name,
            "Best_Alpha": best_alpha,
            "Val_RMSE": final_m["RMSE"],
            "Val_MAE": final_m["MAE"],
            "Val_Bias": final_m["Bias"],
            "Val_CCC": final_m["CCC"],
            "Val_R2": final_m["R2"],
            "Gain_vs_DNN_%": gain_vs_dnn
        })

    # --- ATT Target Transformations ---
    eps = 1e-4
    att_transforms = {
        "Standardized": {
            "forward": lambda y: (y - att_mean) / att_std,
            "inverse": lambda y_hat: np.clip(y_hat * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
        },
        "MinMax_[-1,1]": {
            "forward": lambda y: 2.0 * (y - physics.ATT_MIN) / (physics.ATT_MAX - physics.ATT_MIN) - 1.0,
            "inverse": lambda y_hat: np.clip(physics.ATT_MIN + 0.5 * (y_hat + 1.0) * (physics.ATT_MAX - physics.ATT_MIN), physics.ATT_MIN, physics.ATT_MAX)
        },
        "Raw_Physical": {
            "forward": lambda y: y,
            "inverse": lambda y_hat: np.clip(y_hat, physics.ATT_MIN, physics.ATT_MAX)
        },
        "Bounded_Logit": {
            "forward": lambda y: np.log(np.clip((y - physics.ATT_MIN)/(physics.ATT_MAX - physics.ATT_MIN), eps, 1.0 - eps) / (1.0 - np.clip((y - physics.ATT_MIN)/(physics.ATT_MAX - physics.ATT_MIN), eps, 1.0 - eps))),
            "inverse": lambda y_hat: physics.ATT_MIN + (physics.ATT_MAX - physics.ATT_MIN) / (1.0 + np.exp(-np.clip(y_hat, -15.0, 15.0)))
        }
    }

    for t_name, t_funcs in att_transforms.items():
        print(f"\nEvaluating ATT Target Transform: {t_name}")
        y_tr_t = t_funcs["forward"](Y_train_raw[:, 1])

        seed_cache = []
        for seed in seeds:
            elm = FastGPUChunkedELM(n_hidden=2048, activation="tanh", seed=seed)
            HtH, HtY = elm.compute_HtH_HtY(F_tr_att, y_tr_t)
            seed_cache.append((elm, HtH, HtY))

        best_alpha = None
        best_rmse = float("inf")

        for alpha in alphas:
            rmse_seeds = []
            for elm, HtH, HtY in seed_cache:
                elm.solve_alpha(HtH, HtY, alpha)
                pred_raw = elm.predict(F_val_att)
                pred_phys = t_funcs["inverse"](pred_raw)
                m = calc_metrics(Y_val_raw[:, 1], pred_phys)
                rmse_seeds.append(m["RMSE"])
            avg_rmse = float(np.mean(rmse_seeds))
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_alpha = alpha

        del seed_cache
        torch.cuda.empty_cache()

        preds_5 = []
        for seed in [42, 101, 2024, 7, 99]:
            elm = FastGPUChunkedELM(n_hidden=2048, activation="tanh", seed=seed)
            HtH, HtY = elm.compute_HtH_HtY(F_tr_att, y_tr_t)
            elm.solve_alpha(HtH, HtY, best_alpha)
            preds_5.append(t_funcs["inverse"](elm.predict(F_val_att)))
            del HtH, HtY, elm
            torch.cuda.empty_cache()

        avg_pred = np.mean(preds_5, axis=0)
        final_m = calc_metrics(Y_val_raw[:, 1], avg_pred)
        gain_vs_dnn = (att_dnn_metrics["RMSE"] - final_m["RMSE"]) / att_dnn_metrics["RMSE"] * 100.0
        early_mask = (Y_val_raw[:, 1] >= 0.5) & (Y_val_raw[:, 1] < 1.0)
        early_rmse = float(np.sqrt(np.mean((avg_pred[early_mask] - Y_val_raw[early_mask, 1]) ** 2)))
        print(f"  Best alpha={best_alpha}: Val RMSE={final_m['RMSE']:.4f}, MAE={final_m['MAE']:.4f}, Bias={final_m['Bias']:.4f}, Gain={gain_vs_dnn:+.2f}%, Early RMSE={early_rmse:.4f}")

        target_results.append({
            "Target": "ATT",
            "Transformation": t_name,
            "Best_Alpha": best_alpha,
            "Val_RMSE": final_m["RMSE"],
            "Val_MAE": final_m["MAE"],
            "Val_Bias": final_m["Bias"],
            "Val_CCC": final_m["CCC"],
            "Val_R2": final_m["R2"],
            "Gain_vs_DNN_%": gain_vs_dnn,
            "Early_ATT_RMSE": early_rmse
        })

    df_target = pd.DataFrame(target_results)
    df_target.to_csv(os.path.join(OUT_DIR, "stage2_target_transformations.csv"), index=False)
    print("\nSaved stage2_target_transformations.csv")

    # =========================================================================
    # PART 2: LATENT FEATURE NORMALIZATION & CONDITIONING
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: LATENT FEATURE NORMALIZATION & CONDITIONING")
    print("=" * 80)

    feat_results = []
    # CBF Latent Transformations
    mu_f_cbf = F_tr_cbf.mean(axis=0, keepdims=True)
    std_f_cbf = F_tr_cbf.std(axis=0, keepdims=True) + 1e-8
    med_f_cbf = np.median(F_tr_cbf, axis=0, keepdims=True)
    iqr_f_cbf = np.percentile(F_tr_cbf, 75, axis=0, keepdims=True) - np.percentile(F_tr_cbf, 25, axis=0, keepdims=True) + 1e-8

    f_cbf_dict = {
        "Raw_Latents": (F_tr_cbf, F_val_cbf),
        "Standardized_Latents": ((F_tr_cbf - mu_f_cbf) / std_f_cbf, (F_val_cbf - mu_f_cbf) / std_f_cbf),
        "Robust_Scaled_Latents": ((F_tr_cbf - med_f_cbf) / iqr_f_cbf, (F_val_cbf - med_f_cbf) / iqr_f_cbf)
    }

    y_tr_cbf_std = (Y_train_raw[:, 0] - cbf_mean) / cbf_std
    for f_name, (f_tr, f_val) in f_cbf_dict.items():
        cond_num = float(np.linalg.cond(f_tr[:5000]))
        preds_5 = []
        for seed in [42, 101, 2024, 7, 99]:
            elm = FastGPUChunkedELM(n_hidden=2048, activation="tanh", seed=seed)
            HtH, HtY = elm.compute_HtH_HtY(f_tr, y_tr_cbf_std)
            elm.solve_alpha(HtH, HtY, 10.0)
            pred_std = elm.predict(f_val)
            preds_5.append(np.clip(pred_std * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX))
            del HtH, HtY, elm
            torch.cuda.empty_cache()

        avg_pred = np.mean(preds_5, axis=0)
        m = calc_metrics(Y_val_raw[:, 0], avg_pred)
        gain = (cbf_dnn_metrics["RMSE"] - m["RMSE"]) / cbf_dnn_metrics["RMSE"] * 100.0
        print(f"CBF {f_name}: Cond={cond_num:.1e}, Val RMSE={m['RMSE']:.4f}, Gain={gain:+.2f}%")
        feat_results.append({
            "Target": "CBF",
            "Latent_Scaling": f_name,
            "Condition_Number": cond_num,
            "Val_RMSE": m["RMSE"],
            "Val_MAE": m["MAE"],
            "Val_Bias": m["Bias"],
            "Val_CCC": m["CCC"],
            "Gain_vs_DNN_%": gain
        })

    # ATT Latent Transformations
    mu_f_att = F_tr_att.mean(axis=0, keepdims=True)
    std_f_att = F_tr_att.std(axis=0, keepdims=True) + 1e-8
    med_f_att = np.median(F_tr_att, axis=0, keepdims=True)
    iqr_f_att = np.percentile(F_tr_att, 75, axis=0, keepdims=True) - np.percentile(F_tr_att, 25, axis=0, keepdims=True) + 1e-8

    f_att_dict = {
        "Raw_Latents": (F_tr_att, F_val_att),
        "Standardized_Latents": ((F_tr_att - mu_f_att) / std_f_att, (F_val_att - mu_f_att) / std_f_att),
        "Robust_Scaled_Latents": ((F_tr_att - med_f_att) / iqr_f_att, (F_val_att - med_f_att) / iqr_f_att)
    }

    y_tr_att_std = (Y_train_raw[:, 1] - att_mean) / att_std
    for f_name, (f_tr, f_val) in f_att_dict.items():
        cond_num = float(np.linalg.cond(f_tr[:5000]))
        preds_5 = []
        for seed in [42, 101, 2024, 7, 99]:
            elm = FastGPUChunkedELM(n_hidden=2048, activation="tanh", seed=seed)
            HtH, HtY = elm.compute_HtH_HtY(f_tr, y_tr_att_std)
            elm.solve_alpha(HtH, HtY, 100.0)
            pred_std = elm.predict(f_val)
            preds_5.append(np.clip(pred_std * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX))
            del HtH, HtY, elm
            torch.cuda.empty_cache()

        avg_pred = np.mean(preds_5, axis=0)
        m = calc_metrics(Y_val_raw[:, 1], avg_pred)
        gain = (att_dnn_metrics["RMSE"] - m["RMSE"]) / att_dnn_metrics["RMSE"] * 100.0
        print(f"ATT {f_name}: Cond={cond_num:.1e}, Val RMSE={m['RMSE']:.4f}, Gain={gain:+.2f}%")
        feat_results.append({
            "Target": "ATT",
            "Latent_Scaling": f_name,
            "Condition_Number": cond_num,
            "Val_RMSE": m["RMSE"],
            "Val_MAE": m["MAE"],
            "Val_Bias": m["Bias"],
            "Val_CCC": m["CCC"],
            "Gain_vs_DNN_%": gain
        })

    df_feat = pd.DataFrame(feat_results)
    df_feat.to_csv(os.path.join(OUT_DIR, "stage2_feature_scaling.csv"), index=False)
    print("\nSaved stage2_feature_scaling.csv")

    # =========================================================================
    # PART 3: SIGNAL-BASED DIRECT SCALING (X -> Y)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: DIRECT INPUT SIGNAL SCALING FOR SIGNAL ELM (4 PLDs -> CBF / ATT)")
    print("=" * 80)

    X_med = np.median(X_train_raw, axis=0, keepdims=True)
    X_iqr = np.percentile(X_train_raw, 75, axis=0, keepdims=True) - np.percentile(X_train_raw, 25, axis=0, keepdims=True) + 1e-8
    X_min_val = X_train_raw.min(axis=0, keepdims=True)
    X_max_val = X_train_raw.max(axis=0, keepdims=True)

    sig_scalings = {
        "Standard_ZScore": ((X_train_raw - X_mean) / X_std, (X_val_raw - X_mean) / X_std),
        "Robust_MedianIQR": ((X_train_raw - X_med) / X_iqr, (X_val_raw - X_med) / X_iqr),
        "Log_Scaled_log1p": (np.log1p(X_train_raw), np.log1p(X_val_raw)),
        "MinMax_[0,1]": ((X_train_raw - X_min_val) / (X_max_val - X_min_val + 1e-8), (X_val_raw - X_min_val) / (X_max_val - X_min_val + 1e-8))
    }

    sig_results = []
    for s_name, (x_tr, x_val) in sig_scalings.items():
        s_mean = x_tr.mean(axis=0, keepdims=True)
        s_std = x_tr.std(axis=0, keepdims=True) + 1e-8
        x_tr_norm = (x_tr - s_mean) / s_std
        x_val_norm = (x_val - s_mean) / s_std

        cond_x = float(np.linalg.cond(x_tr_norm[:5000]))

        # CBF Signal ELM
        preds_cbf = []
        for seed in [42, 101, 2024]:
            elm = FastGPUChunkedELM(n_hidden=2048, activation="tanh", seed=seed)
            HtH, HtY = elm.compute_HtH_HtY(x_tr_norm, y_tr_cbf_std)
            elm.solve_alpha(HtH, HtY, 10.0)
            preds_cbf.append(np.clip(elm.predict(x_val_norm) * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX))
            del HtH, HtY, elm
            torch.cuda.empty_cache()

        avg_cbf = np.mean(preds_cbf, axis=0)
        m_cbf = calc_metrics(Y_val_raw[:, 0], avg_cbf)

        # ATT Signal ELM
        preds_att = []
        for seed in [42, 101, 2024]:
            elm = FastGPUChunkedELM(n_hidden=2048, activation="tanh", seed=seed)
            HtH, HtY = elm.compute_HtH_HtY(x_tr_norm, y_tr_att_std)
            elm.solve_alpha(HtH, HtY, 100.0)
            preds_att.append(np.clip(elm.predict(x_val_norm) * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX))
            del HtH, HtY, elm
            torch.cuda.empty_cache()

        avg_att = np.mean(preds_att, axis=0)
        m_att = calc_metrics(Y_val_raw[:, 1], avg_att)

        print(f"Signal ELM ({s_name}): CBF Val RMSE={m_cbf['RMSE']:.4f}, ATT Val RMSE={m_att['RMSE']:.4f}")
        sig_results.append({
            "Signal_Scaling": s_name,
            "Condition_Number": cond_x,
            "CBF_Val_RMSE": m_cbf["RMSE"],
            "CBF_Val_MAE": m_cbf["MAE"],
            "CBF_Val_CCC": m_cbf["CCC"],
            "ATT_Val_RMSE": m_att["RMSE"],
            "ATT_Val_MAE": m_att["MAE"],
            "ATT_Val_CCC": m_att["CCC"]
        })

    df_sig = pd.DataFrame(sig_results)
    df_sig.to_csv(os.path.join(OUT_DIR, "stage2_signal_scaling.csv"), index=False)
    print("\nSaved stage2_signal_scaling.csv")
    print("\nSTAGE 2 EVALUATION FINISHED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
