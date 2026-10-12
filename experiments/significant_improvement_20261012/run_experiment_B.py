"""
run_experiment_B.py
Experiment B: Multi-Task DNN with Shared Signal Encoder and Task-Specific Heads.
Evaluates joint representation learning vs separate networks.
Loss: L_total = w_cbf * MSE(cbf) + w_att * MSE(att)
Predefined validation sweep:
  B1: w_cbf = 1.0, w_att = 1.0 (Equal)
  B2: w_cbf = 2.0, w_att = 1.0 (CBF-Prioritized)
  B3: w_cbf = 0.5, w_att = 1.0 (ATT-Prioritized)
Strict validation evaluation (N=100,000). Test set locked.
Storage on Drive D; host RAM < 80 MB.
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

class MultiTaskASLNet(nn.Module):
    def __init__(self, in_features=4, shared_dim=128, head_dim=64):
        super().__init__()
        # Shared 4-layer representation encoder
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
        # Dedicated CBF Output Head (2 layers)
        self.cbf_head = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1)
        )
        # Dedicated ATT Output Head (2 layers)
        self.att_head = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1)
        )

    def forward(self, x):
        h = self.encoder(x)
        cbf_out = self.cbf_head(h)
        att_out = self.att_head(h)
        return cbf_out, att_out

    def extract_features(self, x):
        return self.encoder(x)

def main():
    print("=" * 80)
    print("EXPERIMENT B: MULTI-TASK DNN (SHARED ENCODER + SEPARATE HEADS)")
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

    # 2. Memory-map Data
    x_tr_mmap = np.load(os.path.join(CACHE_DIR, "X_train.npy"), mmap_mode='r')
    y_tr_mmap = np.load(os.path.join(CACHE_DIR, "Y_train.npy"), mmap_mode='r')
    x_val_mmap = np.load(os.path.join(CACHE_DIR, "X_val.npy"), mmap_mode='r')
    y_val_mmap = np.load(os.path.join(CACHE_DIR, "Y_val.npy"), mmap_mode='r')

    # Load 250k training slice for balanced comparison with Experiment A
    N_tr = 250_000
    rng = np.random.default_rng(2026)
    idx_tr = rng.choice(1_000_000, size=N_tr, replace=False)

    X_tr_n = torch.from_numpy(((x_tr_mmap[idx_tr] - X_mean) / X_std).astype(np.float32)).to(device)
    Y_cbf_tr_std = torch.from_numpy(((y_tr_mmap[idx_tr, 0] - cbf_mean) / cbf_std).astype(np.float32)).unsqueeze(1).to(device)
    Y_att_tr_std = torch.from_numpy(((y_tr_mmap[idx_tr, 1] - att_mean) / att_std).astype(np.float32)).unsqueeze(1).to(device)

    X_val_n = ((x_val_mmap[:] - X_mean) / X_std).astype(np.float32)
    Y_cbf_val = y_val_mmap[:, 0]
    Y_att_val = y_val_mmap[:, 1]
    Y_cbf_val_std = ((Y_cbf_val - cbf_mean) / cbf_std).astype(np.float32)
    Y_att_val_std = ((Y_att_val - att_mean) / att_std).astype(np.float32)

    X_val_t = torch.from_numpy(X_val_n).to(device)
    Y_cbf_val_t = torch.from_numpy(Y_cbf_val_std).unsqueeze(1).to(device)
    Y_att_val_t = torch.from_numpy(Y_att_val_std).unsqueeze(1).to(device)

    configs = [
        ("B1_Equal_Weights", 1.0, 1.0),
        ("B2_CBF_Prioritized", 2.0, 1.0),
        ("B3_ATT_Prioritized", 0.5, 1.0)
    ]

    criterion = nn.MSELoss()
    records = []
    batch_size = 1024
    n_batches = int(np.ceil(N_tr / batch_size))

    for cfg_name, w_cbf, w_att in configs:
        print(f"\n--- Training Multi-Task Config: {cfg_name} (w_CBF={w_cbf}, w_ATT={w_att}) ---")
        model = MultiTaskASLNet(in_features=4, shared_dim=128, head_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_loss = float("inf")
        best_state = None
        t0 = time.time()

        for ep in range(1, 26):
            model.train()
            perm = torch.randperm(N_tr, device=device)
            for b in range(n_batches):
                b_idx = perm[b*batch_size : (b+1)*batch_size]
                optimizer.zero_grad()
                pred_cbf, pred_att = model(X_tr_n[b_idx])
                loss_cbf = criterion(pred_cbf, Y_cbf_tr_std[b_idx])
                loss_att = criterion(pred_att, Y_att_tr_std[b_idx])
                total_loss = w_cbf * loss_cbf + w_att * loss_att
                total_loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                v_cbf, v_att = model(X_val_t)
                val_l_c = criterion(v_cbf, Y_cbf_val_t).item()
                val_l_a = criterion(v_att, Y_att_val_t).item()
                val_loss = w_cbf * val_l_c + w_att * val_l_a

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_time = time.time() - t0
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            pred_cbf_v, pred_att_v = model(X_val_t)
            p_cbf_phys = np.clip(pred_cbf_v.squeeze().cpu().numpy() * cbf_std + cbf_mean, physics.CBF_MIN, physics.CBF_MAX)
            p_att_phys = np.clip(pred_att_v.squeeze().cpu().numpy() * att_std + att_mean, physics.ATT_MIN, physics.ATT_MAX)

        m_cbf = calc_metrics(Y_cbf_val, p_cbf_phys)
        m_att = calc_metrics(Y_att_val, p_att_phys)

        print(f"  Result ({cfg_name}):")
        print(f"    CBF Val RMSE: {m_cbf['RMSE']:.4f}, MAE: {m_cbf['MAE']:.4f}, CCC: {m_cbf['CCC']:.4f}")
        print(f"    ATT Val RMSE: {m_att['RMSE']:.4f} s, MAE: {m_att['MAE']:.4f} s, CCC: {m_att['CCC']:.4f}")
        print(f"    Training Time: {train_time:.1f}s")

        # Save checkpoint
        torch.save(best_state, os.path.join(CKPT_DIR, f"expB_multitask_{cfg_name}.pth"))

        records.append({
            "Experiment": "Experiment B (Multi-Task)",
            "Configuration": cfg_name,
            "w_CBF": w_cbf,
            "w_ATT": w_att,
            "CBF_Val_RMSE": m_cbf["RMSE"],
            "CBF_Val_MAE": m_cbf["MAE"],
            "CBF_Val_CCC": m_cbf["CCC"],
            "CBF_Val_R2": m_cbf["R2"],
            "ATT_Val_RMSE": m_att["RMSE"],
            "ATT_Val_MAE": m_att["MAE"],
            "ATT_Val_CCC": m_att["CCC"],
            "ATT_Val_R2": m_att["R2"],
            "Train_Time_Sec": train_time
        })

    df_b = pd.DataFrame(records)
    csv_out = os.path.join(TABLES_DIR, "experiment_B_val_results.csv")
    df_b.to_csv(csv_out, index=False)
    print(f"\nSaved Experiment B Validation Results to: {csv_out}")
    print(df_b.to_string(index=False))
    print("=" * 80)
    print("EXPERIMENT B COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

