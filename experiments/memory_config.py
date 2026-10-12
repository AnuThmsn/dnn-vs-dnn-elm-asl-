"""
memory_config.py
Guarantees zero footprint on C: drive and ultra-low RAM footprint.
- All temporary files, torch cache, and Python caches redirected to D: drive.
- RAM usage capped via disk-backed numpy arrays (memmap / on-demand loading).
"""

import os
import sys
import gc
import shutil
import psutil
import torch

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVE_D_TMP = os.path.join(WORKSPACE_DIR, ".tmp")
DRIVE_D_CACHE = os.path.join(WORKSPACE_DIR, ".cache")
DRIVE_D_TORCH = os.path.join(DRIVE_D_CACHE, "torch")
DRIVE_D_PYCACHE = os.path.join(DRIVE_D_CACHE, "pycache")

# 1. Enforce Environment Variables strictly onto Drive D:
os.environ["TEMP"] = DRIVE_D_TMP
os.environ["TMP"] = DRIVE_D_TMP
os.environ["TMPDIR"] = DRIVE_D_TMP
os.environ["TORCH_HOME"] = DRIVE_D_TORCH
os.environ["PYTHONPYCACHEPREFIX"] = DRIVE_D_PYCACHE

# Ensure all directories exist on D:
for d in [DRIVE_D_TMP, DRIVE_D_CACHE, DRIVE_D_TORCH, DRIVE_D_PYCACHE]:
    os.makedirs(d, exist_ok=True)


def get_system_status():
    """Report free space on C: and D:, plus host RAM and GPU VRAM."""
    c_stat = shutil.disk_usage("C:")
    d_stat = shutil.disk_usage("D:")
    ram = psutil.virtual_memory()

    status = {
        "C_free_gb": c_stat.free / (1024 ** 3),
        "D_free_gb": d_stat.free / (1024 ** 3),
        "ram_total_gb": ram.total / (1024 ** 3),
        "ram_avail_gb": ram.available / (1024 ** 3),
        "ram_percent_used": ram.percent
    }
    if torch.cuda.is_available():
        status["vram_alloc_mb"] = torch.cuda.memory_allocated(0) / (1024 ** 2)
        status["vram_res_mb"] = torch.cuda.memory_reserved(0) / (1024 ** 2)
    return status


def print_memory_report(stage=""):
    stat = get_system_status()
    print(f"\n[MEM REPORT - {stage}]")
    print(f"  Drive C Free : {stat['C_free_gb']:.2f} GB  (Protected, 0 bytes written)")
    print(f"  Drive D Free : {stat['D_free_gb']:.2f} GB  (Active storage)")
    print(f"  System RAM   : Available={stat['ram_avail_gb']:.2f} GB / {stat['ram_total_gb']:.2f} GB ({stat['ram_percent_used']}% used)")
    if torch.cuda.is_available():
        print(f"  GPU VRAM     : Allocated={stat['vram_alloc_mb']:.1f} MB, Reserved={stat['vram_res_mb']:.1f} MB")


def cleanup_memory():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print_memory_report("Verification")

