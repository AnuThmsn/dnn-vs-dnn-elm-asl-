import subprocess
import sys

def run_cmd(cmd_list):
    print(f"\n>>> Running: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.stdout.strip():
        print("STDOUT:\n", res.stdout.strip())
    if res.stderr.strip():
        print("STDERR:\n", res.stderr.strip())
    print(f"EXIT CODE: {res.returncode}")
    return res.returncode

run_cmd(["git", "init"])
run_cmd(["git", "branch", "-M", "main"])

# Check remote
res = subprocess.run(["git", "remote"], capture_output=True, text=True)
if "origin" in res.stdout.split():
    run_cmd(["git", "remote", "set-url", "origin", "https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-"])
else:
    run_cmd(["git", "remote", "add", "origin", "https://github.com/AnuThmsn/dnn-vs-dnn-elm-asl-"])

run_cmd(["git", "add", "."])
run_cmd(["git", "status", "-s"])

commit_msg = "Initial commit: ASL DNN vs DNN-ELM baseline experiments, physics simulator, dataset generation, and audit tools"
run_cmd(["git", "commit", "-m", commit_msg])

print("\nAttempting git push to origin main...")
ret = run_cmd(["git", "push", "-u", "origin", "main"])
sys.exit(ret)

