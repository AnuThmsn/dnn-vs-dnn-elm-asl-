"""
stage1_architecture_audit.py
Stage 1 Architecture Audit and Non-Invasive Verification for Experiment 4.
Audits the frozen DNN architectures (CBFnet_G_baselineA and ATTnet_G_baselineA),
defines the multi-layer extraction scheme, verifies 100% preservation of forward
model predictions, and outputs config.json.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

# Set workspace paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))

import memory_config
from physics import (
    PLDS, TAU, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX,
    T1T, T1A, ALPHA, BETA, LAMBDA
)
from train_baseline_A import StandardizedNet

OUT_DIR = os.path.join(BASE_DIR, "experiments", "multilayer_elm_04")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_DIR = os.path.join(BASE_DIR, "experiments", "cache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stage 1 Audit] Compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def inspect_and_verify():
    # 1. Load Normalization Statistics
    norm_path = os.path.join(BASE_DIR, "experiments", "baseline_A", "normalization_stats.npz")
    norm_data = np.load(norm_path)
    X_mean = norm_data["X_mean"]
    X_std = norm_data["X_std"]
    y_cbf_mean = float(norm_data["cbf_mean"])
    y_cbf_std = float(norm_data["cbf_std"])
    y_att_mean = float(norm_data["att_mean"])
    y_att_std = float(norm_data["att_std"])

    print(f"[Stage 1 Audit] Loaded normalization statistics:")
    print(f"  X_mean: {X_mean}, X_std: {X_std}")
    print(f"  CBF mean: {y_cbf_mean:.4f}, std: {y_cbf_std:.4f}")
    print(f"  ATT mean: {y_att_mean:.4f}, std: {y_att_std:.4f}")

    # 2. Reconstruct Model Instances
    # CBF net: input=4, n_hidden=5, neurons=128
    cbf_net = StandardizedNet(
        input_dim=4, n_hidden_layers=5, n_neurons=128,
        y_mean=y_cbf_mean, y_std=y_cbf_std, y_min=CBF_MIN, y_max=CBF_MAX
    ).to(device)
    cbf_ckpt_path = os.path.join(BASE_DIR, "experiments", "baseline_A", "CBFnet_G_baselineA.pth")
    cbf_net.load_state_dict(torch.load(cbf_ckpt_path, map_location=device))
    cbf_net.eval()

    # ATT net: input=4, n_hidden=4, neurons=64
    att_net = StandardizedNet(
        input_dim=4, n_hidden_layers=4, n_neurons=64,
        y_mean=y_att_mean, y_std=y_att_std, y_min=ATT_MIN, y_max=ATT_MAX
    ).to(device)
    att_ckpt_path = os.path.join(BASE_DIR, "experiments", "baseline_A", "ATTnet_G_baselineA.pth")
    att_net.load_state_dict(torch.load(att_ckpt_path, map_location=device))
    att_net.eval()

    print("\n--- CBFnet Backbone Structure ---")
    cbf_layers = list(cbf_net.backbone.children())
    for idx, layer in enumerate(cbf_layers):
        print(f"  [{idx}] {layer}")

    print("\n--- ATTnet Backbone Structure ---")
    att_layers = list(att_net.backbone.children())
    for idx, layer in enumerate(att_layers):
        print(f"  [{idx}] {layer}")

    # 3. Define Non-Invasive Multi-Layer Feature Extractor
    class MultiLayerExtractor:
        def __init__(self, net, is_cbf=True):
            self.net = net
            self.is_cbf = is_cbf
            self.layers = list(net.backbone.children())
            if is_cbf:
                # 5 hidden layers: ELU indices are 1, 3, 5, 7, 9
                self.elu_indices = [1, 3, 5, 7, 9]
                self.penultimate_idx = 9
            else:
                # 4 hidden layers: ELU indices are 1, 3, 5, 7
                self.elu_indices = [1, 3, 5, 7]
                self.penultimate_idx = 7

        def extract_all_layers(self, x_tensor):
            """
            Extract all intermediate hidden layer representations (post-ELU).
            Returns dict: {'h1': tensor, 'h2': tensor, ...}
            """
            activations = {}
            h = x_tensor
            h_idx = 1
            for idx, layer in enumerate(self.layers):
                h = layer(h)
                if idx in self.elu_indices:
                    activations[f"h{h_idx}"] = h
                    h_idx += 1
            return activations

        def predict_physical_with_features(self, x_np, X_mean, X_std):
            """
            Standardize input, extract features, and predict physical values.
            Ensures exact mathematical identity with net.predict_physical().
            """
            x_norm = (x_np - X_mean) / X_std
            x_t = torch.tensor(x_norm, dtype=torch.float32, device=device)
            with torch.no_grad():
                feats = self.extract_all_layers(x_t)
                feats_np = {k: v.cpu().numpy() for k, v in feats.items()}
                # Penultimate features check
                penult_orig = self.net.penultimate_features(x_t)
                penult_hook = feats_np[f"h{len(self.elu_indices)}"]
                # Output prediction
                std_pred = self.net.forward(x_t).squeeze(-1)
                phys_pred = std_pred * self.net.y_std + self.net.y_mean
                phys_clamped = phys_pred.clamp(self.net.y_min, self.net.y_max).cpu().numpy()
            return feats_np, phys_clamped, penult_orig

    cbf_extractor = MultiLayerExtractor(cbf_net, is_cbf=True)
    att_extractor = MultiLayerExtractor(att_net, is_cbf=False)

    # 4. Non-Invasive Verification on Validation Data
    print("\n[Stage 1 Audit] Testing on validation subset for exact non-invasive preservation...")
    X_val = np.load(os.path.join(CACHE_DIR, "X_val.npy"))[:1000]
    
    # CBF test
    cbf_orig_pred = cbf_net.predict_physical((X_val - X_mean) / X_std)
    cbf_feats, cbf_hook_pred, cbf_penult_orig = cbf_extractor.predict_physical_with_features(X_val, X_mean, X_std)
    
    cbf_pred_diff = np.max(np.abs(cbf_orig_pred - cbf_hook_pred))
    cbf_penult_diff = np.max(np.abs(cbf_penult_orig - cbf_feats["h5"]))
    
    # ATT test
    att_orig_pred = att_net.predict_physical((X_val - X_mean) / X_std)
    att_feats, att_hook_pred, att_penult_orig = att_extractor.predict_physical_with_features(X_val, X_mean, X_std)
    
    att_pred_diff = np.max(np.abs(att_orig_pred - att_hook_pred))
    att_penult_diff = np.max(np.abs(att_penult_orig - att_feats["h4"]))

    print(f"  CBF Prediction Max Absolute Difference: {cbf_pred_diff:.2e} (Passed: {cbf_pred_diff < 1e-6})")
    print(f"  CBF Penultimate Feature Max Absolute Difference: {cbf_penult_diff:.2e} (Passed: {cbf_penult_diff < 1e-6})")
    print(f"  ATT Prediction Max Absolute Difference: {att_pred_diff:.2e} (Passed: {att_pred_diff < 1e-6})")
    print(f"  ATT Penultimate Feature Max Absolute Difference: {att_penult_diff:.2e} (Passed: {att_penult_diff < 1e-6})")

    assert cbf_pred_diff < 1e-6, "CBF prediction mismatch!"
    assert cbf_penult_diff < 1e-6, "CBF penultimate feature mismatch!"
    assert att_pred_diff < 1e-6, "ATT prediction mismatch!"
    assert att_penult_diff < 1e-6, "ATT penultimate feature mismatch!"

    # 5. Output Configuration
    config = {
        "experiment": "Experiment 4 - Multi-Layer Feature ELM",
        "fixed_acquisition": {
            "plds_seconds": [float(p) for p in PLDS],
            "tau_seconds": float(TAU),
            "noise_model": "dual_channel_rician",
            "reference_constants": {
                "T1_blood": float(T1A),
                "T1_tissue": float(T1T),
                "alpha": float(ALPHA),
                "beta": float(BETA),
                "lambda": float(LAMBDA)
            }
        },
        "target_ranges": {
            "cbf_ml_100g_min": [float(CBF_MIN), float(CBF_MAX)],
            "att_seconds": [float(ATT_MIN), float(ATT_MAX)]
        },
        "cbf_network": {
            "checkpoint": "CBFnet_G_baselineA.pth",
            "input_dim": 4,
            "hidden_layers": 5,
            "units_per_layer": 128,
            "activation": "ELU",
            "penultimate_layer_index": 9,
            "penultimate_dim": 128,
            "layer_indices": {
                "h1": 1, "h2": 3, "h3": 5, "h4": 7, "h5": 9
            },
            "layer_dimensions": {
                "h1": 128, "h2": 128, "h3": 128, "h4": 128, "h5": 128
            },
            "y_mean": y_cbf_mean,
            "y_std": y_cbf_std
        },
        "att_network": {
            "checkpoint": "ATTnet_G_baselineA.pth",
            "input_dim": 4,
            "hidden_layers": 4,
            "units_per_layer": 64,
            "activation": "ELU",
            "penultimate_layer_index": 7,
            "penultimate_dim": 64,
            "layer_indices": {
                "h1": 1, "h2": 3, "h3": 5, "h4": 7
            },
            "layer_dimensions": {
                "h1": 64, "h2": 64, "h3": 64, "h4": 64
            },
            "y_mean": y_att_mean,
            "y_std": y_att_std
        },
        "representations": {
            "Rep_A": "Penultimate layer only (CBF: h5 [128-d], ATT: h4 [64-d]) - Control",
            "Rep_B": "Adjacent hidden layers (CBF: h4 + h5 [256-d], ATT: h3 + h4 [128-d])",
            "Rep_C": "Early + penultimate (CBF: h1 + h5 [256-d], ATT: h1 + h4 [128-d])",
            "Rep_D": "Normalized multi-layer (CBF: h1 + h3 + h5 [384-d] L2-normalized, ATT: h1 + h2 + h4 [192-d] L2-normalized)",
            "Rep_E": "Compact PCA projection of all hidden layers fitted strictly on training data (CBF: 128-d, ATT: 64-d)"
        },
        "nonlinear_readouts": [
            "1. Ridge ELM (Tanh)",
            "2. ELM with alternative activations (ELU, ReLU)",
            "3. Kernel-ELM / Random Fourier Features (RBF kernel)",
            "4. Compact regularized readout"
        ],
        "validation_verification": {
            "cbf_pred_diff_max": float(cbf_pred_diff),
            "att_pred_diff_max": float(att_pred_diff),
            "cbf_penult_diff_max": float(cbf_penult_diff),
            "att_penult_diff_max": float(att_penult_diff),
            "status": "PASSED_NON_INVASIVE_VERIFIED"
        }
    }

    config_path = os.path.join(OUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[Stage 1 Audit] Wrote configuration to {config_path}")

    return config

if __name__ == "__main__":
    inspect_and_verify()
