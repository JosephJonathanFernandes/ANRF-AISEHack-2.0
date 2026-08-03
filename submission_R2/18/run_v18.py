"""
run_v18.py — NB03 + N_AUG=8 + ExtraTrees + XGBoost + Stage A NNLS blend
              + Iterative Sibling Conditioning (Stage C)
No GPU required. CPU runtime ~90 min.

Usage:
  python submission_R2/18/run_v18.py
"""
import json, subprocess, sys, os

NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anrf-v18.ipynb")
SCRIPT   = NOTEBOOK.replace(".ipynb", "_extracted.py")
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def extract(nb_path, script_path):
    print(f"Extracting {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    blocks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        lines = [l for l in src.splitlines(keepends=True) if not l.strip().startswith("!")]
        if lines:
            blocks.append("".join(lines))
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks))
    print(f"Written {script_path}")

if __name__ == "__main__":
    extract(NOTEBOOK, SCRIPT)
    print(f"\nRunning from project root: {ROOT}\n")
    result = subprocess.run([sys.executable, SCRIPT], cwd=ROOT)
    sys.exit(result.returncode)
