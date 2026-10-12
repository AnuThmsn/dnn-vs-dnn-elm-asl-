"""
run_experiment_C.py
Experiment C: Physics-Informed Neural Network (PINN) Training.
Adds a differentiable forward-model consistency term:
  L_total = L_parameter + lambda_physics * L_signal
Uses exact Buxton pCASL kinetic model:
  S_hat = differentiable_pcasl_forward(CBF_hat, ATT_hat)
Predefined validation sweep:
  C0: lambda = 0.0 (Supervised Control)
  C1: lambda = 1e-4
  C2: lambda = 1e-3
  C3: lambda = 1e-2
  C4: lambda = 1e-1
Evaluates parameter metrics and reconstructed signal error separately.
Evaluates strictly on Validation Set (N=100,000). Test set locked.
Storage strictly on Drive D; host RAM < 80 MB.
"""

import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = r"D:\dnn vs dnn elm"
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
BASE_DIR = os.path.join(EXP_DIR, "baseline_A")
CACHE_DIR = os.path.join(EXP_DIR, "cache")
OUT_DIR = os.path.join(EXP_DIR, "significant_improvement_20261012")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

sys.path.append(EXP_DIR)
import physics
from train_baseline_A import calc_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-convert constants to torch tensors on device
PLDS_T = torch.from_numpy(physics.PLDS).float().view(1, 4).to(device)
TAU_VAL = physics.TAU
T1T_VAL = physics.T1T
T1A_VAL = physics.T1A
ALPHA_VAL = physics.ALPHA
BETA_VAL = physics.BETA
LAMBDA_VAL = physics.LAMBDA
SCALE_VAL = physics.SCALE

def differentiable_pcasl_forward(cbf_phys, att_phys):
    """
    Differentiable Buxton pCASL forward signal calculation.
    cbf_phys: (B, 1) in mL / (100g * min)
    att_phys: (B, 1) in seconds
    Returns:
        X_hat_clean: (B, 4) clean expected observed signals (2 * S)
    """
    f_val = cbf_phys / (6000.0 * LAMBDA_VAL)

    t_delay = torch.relu(PLDS_T - att_phys)
    t_end = torch.relu(TAU_VAL + PLDS_T - att_phys)
    decay = torch.exp(-t_delay / T1T_VAL) - torch.exp(-t_end / T1T_VAL)

    coeff = 2.0 * ALPHA_VAL * BETA_VAL * T1T_VAL * (1.0 / LAMBDA_VAL) * f_val * torch.exp(-att_phys / T1A_VAL)
    clean_sig = coeff * decay * SCALE_VAL
    return 2.0 * clean_sig

class PhysicsMultiTaskNet(nn.Module):
    def __init__(self, in_features=4, shared_dim=128, head_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
        )
        self.cbf_head = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1)
        )
        self.att_head = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1)
        )

    def forward(self, x):
        h = self.encoder(x)
        cbf_std = self.cbf_head(h)
        att_std = self.att_head(h)
        return cbf_std, att_std

def main():
    print("=" * 80)
    print("EXPERIMENT C: PHYSICS-INFORMED NEURAL NETWORK (PINN) TRAINING")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Normalization parameters
    norm_data = np.load(os.path.join(BASE_DIR, "normalization_stats.npz"))
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    cbf_mean = float(norm_data["cbf_mean"])
    cbf_std = float(norm_data["cbf_std"])
    att_mean = float(norm_data["att_mean"])
    att_std = float(norm_data["att_std"])

    X_mean_t = torch.from_numpy(X_mean).float().to(device)
    X_std_t = torch.from_numpy(X_std).float().to(device)

    # 2. Memory-map Data
    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    # Load 250k training slice for balanced comparison with Experiment A and B
    N_tr = 250_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    X_tr_raw = torch.from_numpy(x_tr_mmap[idx_tr].astype(np.float32)).to(device)
    X_tr_n = ((X_tr_raw - X_mean_t) / X_std_t)
    Y_cbf_tr_std = torch.from_numpy(((y_tr_mmap[idx_tr, 0] - cbf_mean) / cbf_std).astype(np.float32)).unsqueeze(1).to(device)
    Y_att_tr_std = torch.from_numpy(((y_tr_mmap[idx_tr, 1] - att_mean) / att_std).astype(np.float32)).unsqueeze(1).to(device)

    X_val_raw = torch.from_numpy(x_val_mmap[:].astype(np.float32)).to(device)
    X_val_n_np = ((x_val_mmap[:] - X_mean) / X_std).astype(np.float32)
    X_val_t = torch.from_numpy(X_val_n_np).to(device)
    Y_cbf_val = y_val_mmap[:, 0]
    Y_att_val = y_val_mmap[:, 1]
    Y_cbf_val_std = torch.from_numpy(((Y_cbf_val - cbf_mean) / cbf_std).astype(np.float32)).unsqueeze(1).to(device)
    Y_att_val_std = torch.from_numpy(((Y_att_val - att_mean) / att_std).astype(np.float32)).unsqueeze(1).to(device)

    # Predefined validation sweep
    lambdas = [
        ("C0_Supervised_Control", 0.0),
        ("C1_Physics_1e-4", 1e-4),
        ("C2_Physics_1e-3", 1e-3),
        ("C3_Physics_1e-2", 1e-2),
        ("C4_Physics_1e-1", 1e-1)
    ]

    mse_crit = nn.MSELoss()
    batch_size = 1024
    n_batches = int(np.ceil(N_tr / batch_size))
    records = []

    for cfg_name, lam in lambdas:
        print(f"\n--- Training PINN Config: {cfg_name} (lambda_phys = {lam}) ---")
        model = PhysicsMultiTaskNet(in_features=4, shared_dim=128, head_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_param_loss = float("inf")
        best_state = None
        t0 = time.time()

        for ep in range(1, 26):
            model.train()
            perm = torch.randperm(N_tr, device=device)
            for b in range(n_batches):
                b_idx = perm[b*batch_size : (b+1)*batch_size]
                optimizer.zero_grad()
                pred_cbf_std, pred_att_std = model(X_tr_n[b_idx])
                loss_cbf = mse_crit(pred_cbf_std, Y_cbf_tr_std[b_idx])
                loss_att = mse_crit(pred_att_std, Y_att_tr_std[b_idx])
                l_param = loss_cbf + loss_att

                if lam > 0.0:
                    # Map to physical units with smooth softplus / clamp for stability
                    cbf_phys_pred = torch.clamp(pred_cbf_std * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
                    att_phys_pred = torch.clamp(pred_att_std * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)
                    # Compute differentiable forward reconstructed signal
                    x_hat_clean = differentiable_pcasl_forward(cbf_phys_pred, att_phys_pred)
                    # Normalize signal residual by input std
                    l_signal = torch.mean(((x_hat_clean - X_tr_raw[b_idx]) / X_std_t) ** 2)
                    total_loss = l_param + lam * l_signal
                else:
                    total_loss = l_param

                total_loss.backward()
                optimizer.step()

            # Validation evaluation (strictly parameter loss for early stopping)
            model.eval()
            with torch.no_grad():
                v_cbf, v_att = model(X_val_t)
                val_param_l = (mse_crit(v_cbf, Y_cbf_val_std) + mse_crit(v_att, Y_att_val_std)).item()

            if val_param_l < best_val_param_loss:
                best_val_param_loss = val_param_l
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_time = time.time() - t0
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            v_cbf, v_att = model(X_val_t)
            p_cbf_phys = np.clip(v_cbf.squeeze().cpu().numpy() * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
            p_att_phys = np.clip(v_att.squeeze().cpu().numpy() * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

            # Signal reconstruction error on validation set
            cbf_t_val = torch.from_numpy(p_cbf_phys).float().unsqueeze(1).to(device)
            att_t_val = torch.from_numpy(p_att_phys).float().unsqueeze(1).to(device)
            x_recon_clean = differentiable_pcasl_forward(cbf_t_val, att_t_val)
            sig_recon_rmse = float(torch.sqrt(torch.mean((x_recon_clean - X_val_raw) ** 2)).item())

        m_cbf = calc_metrics(Y_cbf_val, p_cbf_phys)
        m_att = calc_metrics(Y_att_val, p_att_phys)

        print(f"  Result ({cfg_name}):")
        print(f"    CBF Val: RMSE={m_cbf['RMSE']:.4f}, MAE={m_cbf['MAE']:.4f}, Bias={m_cbf['Bias']:+.4f}, CCC={m_cbf['CCC']:.4f}")
        print(f"    ATT Val: RMSE={m_att['RMSE']:.4f} s, MAE={m_att['MAE']:.4f} s, Bias={m_att['Bias']:+.4f} s, CCC={m_att['CCC']:.4f}")
        print(f"    Signal Recon RMSE: {sig_recon_rmse:.2f} (Time: {train_time:.1f}s)")

        # Save checkpoint
        torch.save(best_state, os.path.join(CKPT_DIR, f"expC_pinn_{cfg_name}.pth"))

        records.append({
            "Experiment": "Experiment C (Physics-Informed)",
            "Configuration": cfg_name,
            "Lambda_Physics": lam,
            "CBF_Val_RMSE": m_cbf["RMSE"],
            "CBF_Val_MAE": m_cbf["MAE"],
            "CBF_Val_Bias": m_cbf["Bias"],
            "CBF_Val_CCC": m_cbf["CCC"],
            "ATT_Val_RMSE": m_att["RMSE"],
            "ATT_Val_MAE": m_att["MAE"],
            "ATT_Val_Bias": m_att["Bias"],
            "ATT_Val_CCC": m_att["CCC"],
            "Signal_Recon_RMSE": sig_recon_rmse,
            "Train_Time_Sec": train_time
        })

    df_c = pd.DataFrame(records)
    csv_out = os.path.join(TABLES_DIR, "experiment_C_val_results.csv")
    df_c.to_csv(csv_out, index=False)
    print(f"\nSaved Experiment C Validation Results to: {csv_out}")
    print(df_c.to_string(index=False))
    print("=" * 80)
    print("EXPERIMENT C COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

