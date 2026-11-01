"""Run the packaged evaluation, table generation, and figure generation in order."""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(name):
    command = [sys.executable, str(SCRIPTS / name)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    run("evaluate_models.py")
    run("generate_tables.py")
    run("generate_figures.py")
    with (ROOT / "results" / "execution_log.txt").open("a", encoding="utf-8") as log:
        log.write(
            f"\nPackaged reproduction succeeded at {datetime.now(timezone.utc).isoformat()}.\n"
            "Commands: evaluate_models.py; generate_tables.py; generate_figures.py\n"
            "All commands completed with exit code 0.\n"
        )
