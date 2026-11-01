"""
standalone_inference.py
Production-ready standalone inference engine for ASL MRI CBF and ATT estimation:
- Loads frozen 4x64 DNN-G backbones and trained Residual H-DELM V2 artifacts.
- Accepts raw 4-PLD signal observations [x1, x2, x3, x4] without retraining.
- Returns DNN-G CBF, DNN-G ATT, H-DELM CBF, and H-DELM ATT.
- Verifies exact numerical reproducibility against saved pipeline predictions.
"""

import os
import sys
import numpy as np
import torch

# Paths setup
V2_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(V2_DIR)
ROOT_DIR = os.path.abspath(os.path.join(V2_DIR, "..", ".."))
sys.path.append(ROOT_DIR)

# Reuse authoritative DNN-G model from V1 or local
sys.path.append(os.path.join(ROOT_DIR, "experiments", "paper_reproduction_snr10_15_20"))
from models.dnn_g import DNN_G_Net, device
from models.residual_hdelm_v2 import ResidualHDELMV2

class StandaloneASLEstimator:
    def __init__(
        self,
        cbf_dnn_path=None,
        att_dnn_path=None,
        cbf_elm_path=None,
        att_elm_path=None,
        norm_stats_path=None
    ):
        # Default paths if not specified
        if cbf_dnn_path is None:
            cbf_dnn_path = os.path.join(ROOT_DIR, "experiments", "paper_reproduction_snr10_15_20", "models", "cbf_dnn_g_best.pth")
        if att_dnn_path is None:
            att_dnn_path = os.path.join(ROOT_DIR, "experiments", "paper_reproduction_snr10_15_20", "models", "att_dnn_g_best.pth")
        if cbf_elm_path is None:
            cbf_elm_path = os.path.join(V2_DIR, "models", "cbf_hdelm_v2_artifact.npz")
        if att_elm_path is None:
            att_elm_path = os.path.join(V2_DIR, "models", "att_hdelm_v2_artifact.npz")
        if norm_stats_path is None:
            norm_stats_path = os.path.join(ROOT_DIR, "experiments", "paper_reproduction_snr10_15_20", "results", "input_normalization_stats.npz")

        print("Initializing Standalone ASL Estimator:")
        print(f"  Device: {device}")
        
        # 1. Load input normalization stats
        norm_data = np.load(norm_stats_path)
        self.X_mean = norm_data["X_mean"].astype(np.float32)
        self.X_std = norm_data["X_std"].astype(np.float32)
        print("  Loaded input normalization statistics.")

        # 2. Reconstruct & load frozen CBF DNN
        self.cbf_dnn = DNN_G_Net(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=0.0, y_max=100.0)
        self.cbf_dnn.load_state_dict(torch.load(cbf_dnn_path, map_location=device))
        self.cbf_dnn.to(device)
        self.cbf_dnn.eval()
        print(f"  Loaded CBF DNN-G backbone from {os.path.basename(cbf_dnn_path)}")

        # 3. Reconstruct & load frozen ATT DNN
        self.att_dnn = DNN_G_Net(input_dim=4, n_hidden_layers=4, n_neurons=64, y_min=0.5, y_max=3.0)
        self.att_dnn.load_state_dict(torch.load(att_dnn_path, map_location=device))
        self.att_dnn.to(device)
        self.att_dnn.eval()
        print(f"  Loaded ATT DNN-G backbone from {os.path.basename(att_dnn_path)}")

        # 4. Load CBF H-DELM artifact
        self.cbf_elm = ResidualHDELMV2.load_artifact(cbf_elm_path, dnn_net=self.cbf_dnn)
        print(f"  Loaded CBF H-DELM V2 artifact from {os.path.basename(cbf_elm_path)}")

        # 5. Load ATT H-DELM artifact
        self.att_elm = ResidualHDELMV2.load_artifact(att_elm_path, dnn_net=self.att_dnn)
        print(f"  Loaded ATT H-DELM V2 artifact from {os.path.basename(att_elm_path)}")
        print("Standalone ASL Estimator ready for inference.\n")

    def predict(self, X_raw):
        """
        Accepts raw 4-PLD signal measurements of shape (N, 4).
        Returns a dictionary containing:
          - cbf_dnn: DNN-G estimated CBF (mL/100g/min)
          - att_dnn: DNN-G estimated ATT (s)
          - cbf_hdelm: Proposed Residual H-DELM V2 estimated CBF
          - att_hdelm: Proposed Residual H-DELM V2 estimated ATT
          - cbf_residual: Predicted residual adjustment for CBF
          - att_residual: Predicted residual adjustment for ATT
        """
        X_raw = np.asarray(X_raw, dtype=np.float32)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)
        assert X_raw.shape[1] == 4, f"Expected (N, 4) observations, got {X_raw.shape}"

        # Standardize inputs using training statistics
        X_n = ((X_raw - self.X_mean) / self.X_std).astype(np.float32)

        # H-DELM inference (also returns DNN predictions and residuals)
        cbf_hdelm, cbf_dnn, cbf_res = self.cbf_elm.predict_physical(X_raw, X_n)
        att_hdelm, att_dnn, att_res = self.att_elm.predict_physical(X_raw, X_n)

        return {
            "cbf_dnn": cbf_dnn,
            "att_dnn": att_dnn,
            "cbf_hdelm": cbf_hdelm,
            "att_hdelm": att_hdelm,
            "cbf_residual": cbf_res,
            "att_residual": att_res
        }

def verify_standalone_predictions(
    test_data_path,
    v2_preds_path,
    estimator=None
):
    """
    Verifies that StandaloneASLEstimator produces identical numerical predictions
    to the pipeline predictions stored in v2_preds_path.
    """
    print("=" * 70)
    print("VERIFYING STANDALONE INFERENCE MATCH AGAINST PIPELINE PREDICTIONS")
    print("=" * 70)
    if estimator is None:
        estimator = StandaloneASLEstimator()

    test_data = np.load(test_data_path)
    X_test_raw = test_data["X_test"]

    pipe_data = np.load(v2_preds_path)
    
    preds = estimator.predict(X_test_raw)

    keys_to_compare = [
        ("cbf_dnn", "cbf_dnn"),
        ("att_dnn", "att_dnn"),
        ("cbf_hdelm", "cbf_hdelm_v2"),
        ("att_hdelm", "att_hdelm_v2")
    ]

    all_passed = True
    for pred_key, pipe_key in keys_to_compare:
        p_val = preds[pred_key]
        t_val = pipe_data[pipe_key]
        max_diff = float(np.max(np.abs(p_val - t_val)))
        mean_diff = float(np.mean(np.abs(p_val - t_val)))
        passed = np.allclose(p_val, t_val, atol=1e-5, rtol=1e-5)
        status = "PASSED" if passed else "FAILED"
        print(f"  {pred_key:<12} vs {pipe_key:<15}: Max Diff = {max_diff:.2e}, Mean Diff = {mean_diff:.2e} [{status}]")
        if not passed:
            all_passed = False

    assert all_passed, "Standalone inference predictions DO NOT match pipeline predictions!"
    print("\nSUCCESS: Standalone inference verified with exact numerical equivalence (atol=1e-5).\n")
    return True

if __name__ == "__main__":
    # Smoke test on dummy input
    dummy_signals = np.array([
        [750.0, 620.0, 480.0, 320.0],
        [400.0, 350.0, 260.0, 180.0]
    ], dtype=np.float32)
    print("Smoke test of StandaloneASLEstimator:")
    # Paths will be checked once artifacts are created

