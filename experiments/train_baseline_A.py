"""
train_baseline_A.py
Train, freeze, evaluate, and save Baseline A (Experiment G DNN-G):
- CBFnet: 5 hidden layers x 128 neurons (ELU, MSE loss)
- ATTnet: 4 hidden layers x 64 neurons  (ELU, MSE loss)
- Strictly routes all caches, outputs, and temporary data to Drive D.
- Ultra-low RAM footprint (< 100 MB).
"""

import os
import gc
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Enforce Drive D environment and memory controls
import memory_config
from physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    SNR_VALUES, compute_physical_signals, add_rician_noise
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "baseline_A")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")


# -------------------------------------------------------------
# 1. Exact Model Architecture (Cell 9 of notebook 1)
# -------------------------------------------------------------
class StandardizedNet(nn.Module):
    def __init__(self, input_dim, n_hidden_layers, n_neurons, y_mean, y_std, y_min, y_max):
        super().__init__()
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        layers = [nn.Linear(input_dim, n_neurons), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ELU()]
        layers.append(nn.Linear(n_neurons, 1))
        self.backbone = nn.Sequential(*layers)
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.backbone(x)

    def predict_physical(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            std_pred = self.forward(x).squeeze(-1)
            phys_pred = std_pred * self.y_std + self.y_mean
            return phys_pred.clamp(self.y_min, self.y_max).cpu().numpy()

    def penultimate_features(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            h = x
            for layer in list(self.backbone.children())[:-1]:
                h = layer(h)
            return h.cpu().numpy()


# -------------------------------------------------------------
# 2. Metric Computation (Cell 13 of notebook 1)
# -------------------------------------------------------------
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))
    bias = np.mean(err)

    var_true = np.var(y_true)
    r2 = 1.0 - (mse / var_true) if var_true > 1e-12 else 0.0

    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_p = np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_true + var_p + (mean_t - mean_p) ** 2
    ccc = (2.0 * cov) / denom if denom > 1e-12 else 0.0

    return {
        "RMSE": float(rmse),
        "MSE": float(mse),
        "MAE": float(mae),
        "Bias": float(bias),
        "R2": float(r2),
        "CCC": float(ccc)
    }


# -------------------------------------------------------------
# 3. Training Loop (Cell 11 of notebook 1)
# -------------------------------------------------------------
def train_standardized(net, X_tr_n, Y_col_tr, X_val_n, Y_col_val,
                       batch_size=512, max_epochs=200, patience=10, lr=1e-3, verbose=True):
    Y_tr_std = ((Y_col_tr - net.y_mean) / net.y_std).astype(np.float32)
    Y_val_std = ((Y_col_val - net.y_mean) / net.y_std).astype(np.float32)

    ds_tr = TensorDataset(torch.from_numpy(X_tr_n),
                          torch.from_numpy(Y_tr_std).unsqueeze(1))
    ds_val = TensorDataset(torch.from_numpy(X_val_n),
                           torch.from_numpy(Y_val_std).unsqueeze(1))

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, pin_memory=False)
    ld_val = DataLoader(ds_val, batch_size=batch_size, pin_memory=False)

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    no_imp = 0
    history = []

    t0 = time.time()
    for epoch in range(max_epochs):
        net.train()
        tr = 0.0
        for xb, yb in ld_tr:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optim.zero_grad()
            pred = net(xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            tr += loss.item()
        tr /= len(ld_tr)

        net.eval()
        with torch.no_grad():
            vl = 0.0
            for xb, yb in ld_val:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred = net(xb)
                vl += crit(pred, yb).item()
            vl /= len(ld_val)

        history.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": vl})
        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"  Epoch {epoch+1:3d}: train={tr:.6f} val={vl:.6f}")

        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(net.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    train_time = time.time() - t0
    print(f"Training completed in {train_time:.2f} seconds ({len(history)} epochs)")
    net.load_state_dict(best_state)
    del ds_tr, ds_val, ld_tr, ld_val, optim, crit, Y_tr_std, Y_val_std
    memory_config.cleanup_memory()
    return net, history, train_time


def main():
    print("=" * 80)
    print("PHASE 2: TRAINING BASELINE A (EXPERIMENT G DNN-G)")
    print("Storage strictly on Drive D; host RAM capped < 100 MB")
    print("=" * 80)
    memory_config.print_memory_report("Start of Main")

    # 1. Load Normalization Parameters from Drive D cache
    norm_path = os.path.join(CACHE_DIR, "normalization_stats.npz")
    norm_data = np.load(norm_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    print(f"Loaded normalization stats from Drive D:")
    print(f"  X_mean: {X_mean.squeeze()}")
    print(f"  X_std : {X_std.squeeze()}")
    print(f"  CBF mean/std: {cbf_mean:.4f} / {cbf_std:.4f}")
    print(f"  ATT mean/std: {att_mean:.4f} / {att_std:.4f}")

    # Copy normalization stats to baseline_A directory for self-containment
    np.savez(os.path.join(OUT_DIR, "normalization_stats.npz"),
             X_mean=X_mean, X_std=X_std,
             cbf_mean=cbf_mean, cbf_std=cbf_std,
             att_mean=att_mean, att_std=att_std)

    # 2. Memory-map Training and Validation Arrays from Drive D
    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    # Create normalized input tensors (only 16 MB for Train, 1.6 MB for Val)
    print("\nPreparing normalized training inputs (16 MB in RAM)...")
    X_tr_n = ((x_tr_mmap[:] - X_mean) / X_std).astype(np.float32)
    X_val_n = ((x_val_mmap[:] - X_mean) / X_std).astype(np.float32)

    # 3. Model Architecture Initialization
    INPUT_DIM = len(PLDS)
    ATTnet_G = StandardizedNet(INPUT_DIM, n_hidden_layers=4, n_neurons=64,
                               y_mean=att_mean, y_std=att_std, y_min=ATT_MIN, y_max=ATT_MAX).to(device)
    CBFnet_G = StandardizedNet(INPUT_DIM, n_hidden_layers=5, n_neurons=128,
                               y_mean=cbf_mean, y_std=cbf_std, y_min=CBF_MIN, y_max=CBF_MAX).to(device)

    print(f"\nInitialized Baseline A architectures:")
    print(f"  ATTnet_G (4x64)  params: {sum(p.numel() for p in ATTnet_G.parameters()):,}")
    print(f"  CBFnet_G (5x128) params: {sum(p.numel() for p in CBFnet_G.parameters()):,}")

    # 4. Train ATTnet (4x64, MSE loss)
    print("\n" + "=" * 60)
    print("Training ATTnet_G (4 hidden layers x 64 neurons, MSE loss)...")
    print("=" * 60)
    Y_att_tr = y_tr_mmap[:, 1]
    Y_att_val = y_val_mmap[:, 1]
    ATTnet_G, att_hist, att_time = train_standardized(ATTnet_G, X_tr_n, Y_att_tr, X_val_n, Y_att_val,
                                                      lr=1e-3, max_epochs=200, patience=10)
    del Y_att_tr, Y_att_val
    memory_config.cleanup_memory()

    # 5. Train CBFnet (5x128, MSE loss)
    print("\n" + "=" * 60)
    print("Training CBFnet_G (5 hidden layers x 128 neurons, MSE loss)...")
    print("=" * 60)
    Y_cbf_tr = y_tr_mmap[:, 0]
    Y_cbf_val = y_val_mmap[:, 0]
    CBFnet_G, cbf_hist, cbf_time = train_standardized(CBFnet_G, X_tr_n, Y_cbf_tr, X_val_n, Y_cbf_val,
                                                      lr=1e-3, max_epochs=200, patience=10)
    del Y_cbf_tr, Y_cbf_val, X_tr_n, X_val_n
    memory_config.cleanup_memory()

    # 6. Save Checkpoints to Drive D
    att_ckpt = os.path.join(OUT_DIR, "ATTnet_G_baselineA.pth")
    cbf_ckpt = os.path.join(OUT_DIR, "CBFnet_G_baselineA.pth")
    torch.save(ATTnet_G.state_dict(), att_ckpt)
    torch.save(CBFnet_G.state_dict(), cbf_ckpt)
    print(f"\nSaved checkpoints to Drive D:\n  {att_ckpt}\n  {cbf_ckpt}")

    # Save training histories
    with open(os.path.join(OUT_DIR, "training_history.json"), "w") as f:
        json.dump({"att_history": att_hist, "cbf_history": cbf_hist,
                   "att_time_sec": att_time, "cbf_time_sec": cbf_time}, f, indent=2)

    # 7. Evaluation
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE A")
    print("=" * 60)
    ATTnet_G.eval(); CBFnet_G.eval()

    # A. Clean Population Evaluation (2,392 points)
    clean_data = np.load(os.path.join(CACHE_DIR, "clean_grid.npz"))
    X_clean = clean_data["X_clean"]
    cbf_flat = clean_data["cbf_flat"]
    att_flat = clean_data["att_flat"]
    clean_sig = clean_data["clean_sig"]

    X_clean_n = ((X_clean - X_mean) / X_std).astype(np.float32)
    cbf_clean_pred = CBFnet_G.predict_physical(X_clean_n)
    att_clean_pred = ATTnet_G.predict_physical(X_clean_n)

    m_clean_cbf = calc_metrics(cbf_flat, cbf_clean_pred)
    m_clean_att = calc_metrics(att_flat, att_clean_pred)

    print("\n--- Clean Grid (2,392 points) ---")
    print(f"CBF RMSE: {m_clean_cbf['RMSE']:.4f} | MAE: {m_clean_cbf['MAE']:.4f} | R2: {m_clean_cbf['R2']:.4f} | CCC: {m_clean_cbf['CCC']:.4f}")
    print(f"ATT RMSE: {m_clean_att['RMSE']:.4f} | MAE: {m_clean_att['MAE']:.4f} | R2: {m_clean_att['R2']:.4f} | CCC: {m_clean_att['CCC']:.4f}")

    # B. Held-Out Test Set (100,000 samples)
    x_test_mmap = np.load(os.path.join(CACHE_DIR, "X_test.npy"), mmap_mode='r')
    y_test_mmap = np.load(os.path.join(CACHE_DIR, "Y_test.npy"), mmap_mode='r')
    X_test_n = ((x_test_mmap[:] - X_mean) / X_std).astype(np.float32)

    cbf_test_pred = CBFnet_G.predict_physical(X_test_n)
    att_test_pred = ATTnet_G.predict_physical(X_test_n)

    m_test_cbf = calc_metrics(y_test_mmap[:, 0], cbf_test_pred)
    m_test_att = calc_metrics(y_test_mmap[:, 1], att_test_pred)

    print("\n--- Held-Out Test Set (100,000 samples) ---")
    print(f"CBF RMSE: {m_test_cbf['RMSE']:.4f} | MAE: {m_test_cbf['MAE']:.4f} | Bias: {m_test_cbf['Bias']:+.4f} | R2: {m_test_cbf['R2']:.4f} | CCC: {m_test_cbf['CCC']:.4f}")
    print(f"ATT RMSE: {m_test_att['RMSE']:.4f} | MAE: {m_test_att['MAE']:.4f} | Bias: {m_test_att['Bias']:+.4f} | R2: {m_test_att['R2']:.4f} | CCC: {m_test_att['CCC']:.4f}")

    # C. Fixed SNR Robustness Sweep (2,392 points at each SNR)
    print("\n--- Fixed SNR Robustness Grid Sweep ---")
    snr_rows = []
    rng_snr = np.random.default_rng(2026)
    for snr_val in SNR_VALUES:
        X_snr = add_rician_noise(clean_sig, snr_val, rng_snr)
        X_snr_n = ((X_snr - X_mean) / X_std).astype(np.float32)
        cbf_p = CBFnet_G.predict_physical(X_snr_n)
        att_p = ATTnet_G.predict_physical(X_snr_n)
        mc = calc_metrics(cbf_flat, cbf_p)
        ma = calc_metrics(att_flat, att_p)
        snr_rows.append({
            "SNR": snr_val if not np.isinf(snr_val) else "inf",
            "CBF_RMSE": mc["RMSE"], "CBF_MAE": mc["MAE"], "CBF_CCC": mc["CCC"],
            "ATT_RMSE": ma["RMSE"], "ATT_MAE": ma["MAE"], "ATT_CCC": ma["CCC"]
        })

    snr_df = pd.DataFrame(snr_rows)
    print(snr_df.to_string(index=False))

    # Save evaluation tables
    snr_df.to_csv(os.path.join(OUT_DIR, "snr_evaluation_baselineA.csv"), index=False)
    clean_df = pd.DataFrame([{"Target": "CBF", **m_clean_cbf}, {"Target": "ATT", **m_clean_att}])
    clean_df.to_csv(os.path.join(OUT_DIR, "clean_grid_baselineA.csv"), index=False)
    test_df = pd.DataFrame([{"Target": "CBF", **m_test_cbf}, {"Target": "ATT", **m_test_att}])
    test_df.to_csv(os.path.join(OUT_DIR, "test_summary_baselineA.csv"), index=False)

    # Save test predictions for paired bootstrap comparisons in Phase 6
    np.savez(os.path.join(OUT_DIR, "baselineA_test_predictions.npz"),
             cbf_test_pred=cbf_test_pred, att_test_pred=att_test_pred,
             cbf_clean_pred=cbf_clean_pred, att_clean_pred=att_clean_pred)

    memory_config.print_memory_report("After Baseline A Completion")
    print("\n" + "=" * 80)
    print("BASELINE A REPRODUCTION SUCCESSFULLY COMPLETED AND FROZEN ON DRIVE D")
    print("=" * 80)


if __name__ == "__main__":
    main()
