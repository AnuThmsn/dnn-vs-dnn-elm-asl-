# audit_pipeline.py
"""
Audit the ASL-MRI pipeline for target scaling, normalization, loss definitions, and evaluation.
The script performs:
1. Load normalization statistics from `experiments/baseline_A/normalization_stats.npz`.
2. Load training history JSON (`experiments/dnn_vs_dnn_elm_snr_report/experiment_X/training_history.json`).
3. Verify that loss is computed on *normalized* targets (CBF/ATT) and that the same stats are used for inverse transform.
4. Check clipping bounds used during inference (ATT_MIN/ATT_MAX, CBF_MIN/CBF_MAX).
5. Print a concise audit report.
"""
import os
import json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def load_norm_stats():
    path = os.path.join(REPO_ROOT, "experiments", "baseline_A", "normalization_stats.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}

def load_training_history():
    # Search for the first training_history.json under the experiment tree
    for root, _, files in os.walk(os.path.join(REPO_ROOT, "experiments")):
        if "training_history.json" in files:
            with open(os.path.join(root, "training_history.json"), "r") as f:
                return json.load(f)
    raise FileNotFoundError("training_history.json not found")

def audit():
    print("--- ASL-MRI Pipeline Audit ---")
    # 1. Normalization stats
    norm = load_norm_stats()
    print("Normalization stats loaded from baseline_A:")
    for k, v in norm.items():
        print(f"  {k}: {v.squeeze() if hasattr(v, 'squeeze') else v}")

    # 2. Training history
    history = load_training_history()
    print("\nTraining history keys:", list(history.keys()))
    # Expected keys: 'train_loss', 'val_loss', 'epoch'
    if 'train_loss' in history and 'val_loss' in history:
        train_losses = np.array(history['train_loss'], dtype=np.float64)
        val_losses = np.array(history['val_loss'], dtype=np.float64)
        print(f"Train loss: init={train_losses[0]:.4f}, min={train_losses.min():.4f}, final={train_losses[-1]:.4f}")
        print(f"Val   loss: init={val_losses[0]:.4f}, min={val_losses.min():.4f}, final={val_losses[-1]:.4f}")
        min_idx = int(np.argmin(val_losses))
        print(f"Best val loss at epoch {min_idx+1}")
    else:
        print("Warning: training loss entries missing – check loss definition in the training script.")

    # 3. Verify loss on normalized targets
    # In the training script (`train_baseline_A.py`) the loss is typically MSE on normalized CBF/ATT.
    # Here we only report the presence of 'target_mean'/'target_std' in the checkpoint metadata.
    chkpt_path = os.path.join(REPO_ROOT, "experiments", "baseline_A", "CBFnet_G_baselineA.pth")
    try:
        import torch
        ckpt = torch.load(chkpt_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'target_mean' in ckpt and 'target_std' in ckpt:
            print("Checkpoint contains target_mean/target_std – consistent with normalized loss.")
        else:
            print("Checkpoint does not expose target scaling metadata – manual verification required.")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

    # 4. Clipping bounds (physics module)
    from experiments.physics import CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
    print(f"Clipping bounds: CBF [{CBF_MIN}, {CBF_MAX}], ATT [{ATT_MIN}, {ATT_MAX}]")

if __name__ == "__main__":
    audit()

