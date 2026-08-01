"""
Build v14 notebook: NB03 (anrf-app3) with two targeted Stage B improvements:
  1. N_AUG 4 -> 8  (more masked-sibling augmentation = more robust stacker)
  2. Add ExtraTreesRegressor to stage_b_models zoo (NNLS auto-weights it)
"""
import json, os

IN  = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\Shivesh\anrf-app3.ipynb"
OUT = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\14\anrf-v14.ipynb"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(IN, encoding="utf-8") as f:
    nb = json.load(f)

aug_done = False
zoo_done = False
import_done = False

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    src_joined = "".join(src)

    # Identify the NB03 setup cell (has scipy + lgb + N_AUG)
    is_setup = "import scipy.optimize" in src_joined and "import lightgbm" in src_joined

    if is_setup:
        new_src = []
        for line in src:
            # 1. Inject ExtraTrees import right after lgb import
            new_src.append(line)
            if "import lightgbm as lgb" in line and not import_done:
                new_src.append("from sklearn.ensemble import ExtraTreesRegressor\n")
                import_done = True
            # 2. Bump N_AUG
            if "N_AUG = 4" in line and not aug_done:
                new_src[-1] = line.replace("N_AUG = 4", "N_AUG = 8")
                aug_done = True
        cell["source"] = new_src

    # 3. Add ExtraTreesRegressor to stage_b_models zoo
    if not zoo_done and "def stage_b_models" in src_joined:
        new_src = []
        for line in src:
            new_src.append(line)
            if 'RidgeCV(alphas=' in line:
                new_src.append(
                    '        "et": lambda: ExtraTreesRegressor(\n'
                    '            n_estimators=400, max_features=0.5, min_samples_leaf=2,\n'
                    '            n_jobs=-1, random_state=SEED),\n'
                )
                zoo_done = True
        cell["source"] = new_src

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {OUT}")
print(f"  import_done={import_done}  aug_done={aug_done}  zoo_done={zoo_done}")
