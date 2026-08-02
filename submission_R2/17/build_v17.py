"""
build_v17.py — NB03 + Stage B: N_AUG=8, LGB+Ridge+ET+XGB zoo
              + Stage A: NNLS-optimal blend (instead of fixed 50/50)

Built on top of Shivesh's anrf-app3.ipynb (the proven 0.899 baseline).
v14 (0.901) = N_AUG=8 + ExtraTrees
v17         = v14 + XGBoost in zoo + Stage A NNLS blend
"""
import json, os

IN  = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\Shivesh\anrf-app3.ipynb"
OUT = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\17\anrf-v17.ipynb"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(IN, encoding="utf-8") as f:
    nb = json.load(f)

setup_done = False
stageA_done = False
zoo_done = False
import_done = False

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    src_joined = "".join(src)

    # ── NB03 setup cell (scipy + lgb + N_AUG) ────────────────────────────────
    is_setup = "import scipy.optimize" in src_joined and "import lightgbm" in src_joined
    if is_setup and not setup_done:
        new_src = []
        for line in src:
            new_src.append(line)
            if "import lightgbm as lgb" in line and not import_done:
                new_src.append("import xgboost as xgb\n")
                new_src.append("from sklearn.ensemble import ExtraTreesRegressor\n")
                import_done = True
            if "N_AUG = 4" in line:
                new_src[-1] = line.replace("N_AUG = 4", "N_AUG = 8")
        cell["source"] = new_src
        setup_done = True

    # ── Stage A: replace fixed 50/50 blend with NNLS optimal blend ───────────
    # Original: oof[b] = (0.5 * p1 + 0.5 * p2) * sd + mu
    # New: accumulate OOF per model, then NNLS blend
    if not stageA_done and "def build_stage_a_compact" in src_joined:
        new_src = []
        for line in src:
            # Replace the fixed 50/50 OOF accumulation with per-model storage
            if "oof[b] = (0.5 * p1 + 0.5 * p2) * sd + mu" in line:
                indent = len(line) - len(line.lstrip())
                sp = " " * indent
                new_src.append(f"{sp}oof_p1[b] = p1\n")
                new_src.append(f"{sp}oof_p2[b] = p2\n")
            # Replace final test blend (0.5 * f1 + 0.5 * f2)
            elif "out[t] = (0.5 * f1 + 0.5 * f2) * sd + mu" in line:
                indent = len(line) - len(line.lstrip())
                sp = " " * indent
                new_src.append(f"{sp}# NNLS optimal blend weight for this target\n")
                new_src.append(f"{sp}P_oof = np.column_stack([oof_p1, oof_p2])\n")
                new_src.append(f"{sp}import scipy.optimize as _sopt\n")
                new_src.append(f"{sp}w_sa, _ = _sopt.nnls(P_oof * sd + mu, y)\n")
                new_src.append(f"{sp}w_sa = w_sa / w_sa.sum() if w_sa.sum() > 1e-9 else np.array([0.5, 0.5])\n")
                new_src.append(f"{sp}out[t] = (w_sa[0] * f1 + w_sa[1] * f2) * sd + mu\n")
            # Replace final OOF override with NNLS-blended value
            elif "out.loc[s.smiles.values, t] = oof" in line and "# OOF overrides" in "".join(src):
                indent = len(line) - len(line.lstrip())
                sp = " " * indent
                new_src.append(f"{sp}oof_blended = (oof_p1 * w_sa[0] + oof_p2 * w_sa[1]) * sd + mu\n")
                new_src.append(f"{sp}out.loc[s.smiles.values, t] = oof_blended\n")
                new_src.append(f"{sp}print(f\"  stageA {{t:>4}}  n={{len(y):<5}} oof R2={{r2(y, oof_blended):.4f}}  w_lgb={{w_sa[0]:.2f}} w_kr={{w_sa[1]:.2f}}\")\n")
            # Initialise per-model OOF arrays after oof = np.full(...)
            elif "oof = np.full(len(y), np.nan)" in line:
                new_src.append(line)
                indent = len(line) - len(line.lstrip())
                sp = " " * indent
                new_src.append(f"{sp}oof_p1 = np.full(len(y), np.nan)\n")
                new_src.append(f"{sp}oof_p2 = np.full(len(y), np.nan)\n")
            # Remove the old print statement for stageA (we print inside the new block)
            elif 'print(f"  stageA' in line:
                pass  # skip old print; new one is in the oof override block above
            else:
                new_src.append(line)
        cell["source"] = new_src
        stageA_done = True

    # ── Stage B model zoo: add XGBoost ───────────────────────────────────────
    if not zoo_done and "def stage_b_models" in src_joined:
        new_src = []
        for line in src:
            new_src.append(line)
            if 'RidgeCV(alphas=' in line:
                new_src.append(
                    '        "et": lambda: ExtraTreesRegressor(\n'
                    '            n_estimators=400, max_features=0.5, min_samples_leaf=2,\n'
                    '            n_jobs=-1, random_state=SEED),\n'
                    '        "xgb": lambda: xgb.XGBRegressor(\n'
                    '            n_estimators=500 if small else 900, learning_rate=0.03,\n'
                    '            max_depth=3 if small else 6, subsample=0.8,\n'
                    '            colsample_bytree=0.5, reg_lambda=2.0,\n'
                    '            tree_method="hist", n_jobs=-1, random_state=SEED,\n'
                    '            verbosity=0),\n'
                )
                zoo_done = True
        cell["source"] = new_src

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {OUT}")
print(f"  setup_done={setup_done}  import_done={import_done}")
print(f"  stageA_done={stageA_done}  zoo_done={zoo_done}")
