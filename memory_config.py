# Simple memory configuration utilities for low‑RAM experiments
# This stub provides the minimal API used throughout the repository.
# It is deliberately lightweight – only printing messages and invoking
# Python's garbage collector. Real projects may implement more advanced
# memory‑tracking or platform‑specific cleanup.

import gc
import psutil
import os

def print_memory_report(context: str = "") -> None:
    """Print a brief memory usage report.

    Parameters
    ----------
    context: str, optional
        Identifier displayed with the report to help locate the call site.
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[MemoryReport] {context}: {mem_mb:.2f} MB used")

def cleanup_memory() -> None:
    """Force garbage collection and optionally log the freed memory."""
    gc.collect()
    # Optionally print after cleanup for debugging
    print("[MemoryCleanup] garbage collected")

