"""
run_inference.py
Standalone inference module for paper reproduction:
Loads saved 4x64 DNN-G baseline and Residual H-DELM models,
takes any test input dataset, and outputs predictions for CBF and ATT.
"""

import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(EXP_DIR)

from models.dnn_g import DNN_G_Net, device
from models.residual_hdelm import ResidualHDELM

def run_evaluation_inference(X_test_raw, models_dir=None, norm_path=None):
    if models_dir is None:
        models_dir = os.path.join(EXP_DIR, "models")
    if norm_path is None:
        norm_path = os.path.join(EXP_DIR, "results", "input_normalization_stats.npz")

    norm = np.load(norm_path)
    X_mean = norm["X_mean"]
    X_std = norm["X_std"]

    X_test_n = ((X_test_raw - X_mean) / X_std).astype(np.float32)

    # 1. Load DNN-G CBF & ATT
    cbf_dnn = DNN_G_Net(4, 4, 64, 0.0, 100.0).to(device)
    cbf_dnn.load_state_dict(torch.load(os.path.join(models_dir, "cbf_dnn_g_best.pth"), map_location=device))
    cbf_dnn.eval()

    att_dnn = DNN_G_Net(4, 4, 64, 0.5, 3.0).to(device)
    att_dnn.load_state_dict(torch.load(os.path.join(models_dir, "att_dnn_g_best.pth"), map_location=device))
    att_dnn.eval()

    # 2. Predict DNN
    cbf_pred_dnn = cbf_dnn.predict_physical(X_test_n)
    att_pred_dnn = att_dnn.predict_physical(X_test_n)

    return {
        "cbf_dnn": cbf_pred_dnn,
        "att_dnn": att_pred_dnn
    }

