"""
dnn_g.py
Authoritative DNN-G (Baseline) implementation strictly adhering to the paper:
- Exactly 4 hidden layers x 64 neurons for BOTH CBF and ATT networks
- Activation: ELU
- Output transformation:
    CBF = 0.0 + (100.0 - 0.0) * sigmoid(z)
    ATT = 0.5 + (3.0 - 0.5) * sigmoid(z)
- Loss: Normalized MSE: MSE(sigmoid(z), y_norm) where y_norm = (y - y_min)/(y_max - y_min)
- Optimization: Adam (lr=1e-3), batch size 512, max epochs 200, gradient clipping 1.0, patience 10
- Restores best validation checkpoint
- Extracts 64-dimensional penultimate representation
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNN_G_Net(nn.Module):
    def __init__(self, input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=0.0, y_max=100.0):
        super().__init__()
        self.input_dim = input_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.y_range = self.y_max - self.y_min

        # Feature extractor: 4 layers x 64 neurons with ELU
        layers = [nn.Linear(input_dim, n_neurons), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ELU()]
        self.feature_extractor = nn.Sequential(*layers)

        # Output projection to logit z
        self.output_head = nn.Linear(n_neurons, 1)

        # Kaiming normal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.feature_extractor(x)
        z = self.output_head(h)
        return z

    def forward_normalized(self, x):
        """Returns sigmoid(z) in [0, 1]."""
        z = self.forward(x)
        return torch.sigmoid(z)

    def predict_physical(self, x_np, batch_size=10000):
        """Predicts physical output y = y_min + (y_max - y_min) * sigmoid(z)."""
        self.eval()
        N = x_np.shape[0]
        preds = np.empty(N, dtype=np.float32)
        n_batches = int(np.ceil(N / batch_size))

        with torch.no_grad():
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, N)
                xb = torch.from_numpy(x_np[start:end].astype(np.float32)).to(device)
                z = self.forward(xb).squeeze(-1)
                sig = torch.sigmoid(z)
                phys = self.y_min + self.y_range * sig
                preds[start:end] = phys.cpu().numpy()
        return preds

    def penultimate_features(self, x_np, batch_size=10000):
        """Extracts 64-dimensional post-ELU activations of the 4th hidden layer."""
        self.eval()
        N = x_np.shape[0]
        feats = np.empty((N, self.n_neurons), dtype=np.float32)
        n_batches = int(np.ceil(N / batch_size))

        with torch.no_grad():
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, N)
                xb = torch.from_numpy(x_np[start:end].astype(np.float32)).to(device)
                h = self.feature_extractor(xb)
                feats[start:end] = h.cpu().numpy()
        return feats

def train_dnn_g(net, X_tr_n, Y_tr_phys, X_val_n, Y_val_phys,
                batch_size=512, max_epochs=200, patience=10, lr=1e-3,
                log_prefix=""):
    """
    Trains DNN_G_Net with normalized MSE loss:
    y_norm = (y_phys - y_min) / (y_max - y_min)
    pred_norm = sigmoid(z)
    loss = MSE(pred_norm, y_norm)
    """
    net = net.to(device)
    y_min, y_max = net.y_min, net.y_max
    y_range = y_max - y_min

    # Normalize targets to [0, 1]
    Y_tr_norm = ((Y_tr_phys - y_min) / y_range).astype(np.float32)
    Y_val_norm = ((Y_val_phys - y_min) / y_range).astype(np.float32)

    ds_tr = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr_norm).unsqueeze(1))
    ds_val = TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(Y_val_norm).unsqueeze(1))

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, pin_memory=False)
    ld_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    no_imp = 0
    history = []

    t0 = time.time()
    for epoch in range(max_epochs):
        net.train()
        tr_loss = 0.0
        for xb, yb in ld_tr:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = net.forward_normalized(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(ld_tr)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in ld_val:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred = net.forward_normalized(xb)
                val_loss += criterion(pred, yb).item()
            val_loss /= len(ld_val)

        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_rmse_physical": np.sqrt(val_loss) * y_range
        })

        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(f"{log_prefix}Epoch {epoch+1:3d}: train_loss={tr_loss:.6f}, val_loss={val_loss:.6f} (val RMSE={np.sqrt(val_loss)*y_range:.4f})", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(net.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"{log_prefix}Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).", flush=True)
                break

    train_time = time.time() - t0
    net.load_state_dict(best_state)
    print(f"{log_prefix}Training finished in {train_time:.2f}s ({len(history)} epochs). Best Val Loss: {best_val_loss:.6f} (RMSE: {np.sqrt(best_val_loss)*y_range:.4f})", flush=True)
    return net, history, train_time, best_val_loss

