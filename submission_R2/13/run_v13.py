"""
run_v13.py — Extracts and runs the v13 notebook locally.

Requirements:
  - pip install torch lightgbm scikit-learn rdkit pandas numpy scipy
  - GPU recommended (PyTorch will fall back to CPU if not available,
    but Stage A training will be MUCH slower — ~30-60 mins on CPU vs ~8 mins on GPU)

Usage:
  python submission_R2/13/run_v13.py
"""
import json
import subprocess
import sys
import os

NOTEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anrf-v13-nn-sibling.ipynb")
SCRIPT   = NOTEBOOK.replace(".ipynb", "_extracted.py")
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def extract(nb_path, script_path):
    print(f"Extracting code from {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    blocks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        # Skip shell magic lines like !pip install ...
        lines = [l for l in src.splitlines(keepends=True) if not l.strip().startswith("!")]
        if lines:
            blocks.append("".join(lines))

    full = "\n\n".join(blocks)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"Written {script_path}")

if __name__ == "__main__":
    extract(NOTEBOOK, SCRIPT)
    print(f"\nRunning from project root: {ROOT}")
    print("(GPU will be used if available, otherwise CPU — Stage A may take longer)\n")
    result = subprocess.run([sys.executable, SCRIPT], cwd=ROOT)
    sys.exit(result.returncode)
