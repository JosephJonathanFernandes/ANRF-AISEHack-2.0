"""
build_v20.py — Mega Zoo (LGB+Ridge+ET+XGB+CatBoost) + NNLS Stage A
NO Pseudo-Labeling.
"""
import json, os

IN  = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\Shivesh\anrf-app3.ipynb"
OUT = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\20\anrf-v20.ipynb"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(IN, encoding="utf-8") as f:
    nb = json.load(f)

setup_done = False
stageA_done = False
zoo_done = False

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    src_joined = "".join(src)

    if "import scipy.optimize" in src_joined and "import lightgbm" in src_joined and not setup_done:
        new_src = []
        import_done = False
        for line in src:
            new_src.append(line)
            if "import lightgbm as lgb" in line and not import_done:
                new_src.append("import xgboost as xgb\n")
                new_src.append("from catboost import CatBoostRegressor\n")
                new_src.append("from sklearn.ensemble import ExtraTreesRegressor\n")
                import_done = True
            if "N_AUG = 4" in line:
                new_src[-1] = line.replace("N_AUG = 4", "N_AUG = 8")
        cell["source"] = new_src
        setup_done = True

    elif not stageA_done and "def build_stage_a_compact" in src_joined:
        new_src = []
        for line in src:
            if "oof[b] = (0.5 * p1 + 0.5 * p2) * sd + mu" in line:
                indent = len(line) - len(line.lstrip())
                sp = " " * indent
                new_src.append(f"{sp}oof_p1[b] = p1\n")
                new_src.append(f"{sp}oof_p2[b] = p2\n")
            elif "out[t] = (0.5 * f1 + 0.5 * f2) * sd + mu" in line:
                indent = len(line) - len(line.lstrip())
                sp = " " * indent
                new_src.append(f"{sp}P_oof = np.column_stack([oof_p1, oof_p2])\n")
                new_src.append(f"{sp}import scipy.optimize as _sopt\n")
                new_src.append(f"{sp}w_sa, _ = _sopt.nnls(P_oof * sd + mu, y)\n")
                new_src.append(f"{sp}w_sa = w_sa / w_sa.sum() if w_sa.sum() > 1e-9 else np.array([0.5, 0.5])\n")
                new_src.append(f"{sp}out[t] = (w_sa[0] * f1 + w_sa[1] * f2) * sd + mu\n")
            elif "out.loc[s.smiles.values, t] = oof" in line and "# OOF overrides" in src_joined:
                indent = len(line) - len(line.lstrip())
                sp = " " * indent
                new_src.append(f"{sp}oof_blended = (oof_p1 * w_sa[0] + oof_p2 * w_sa[1]) * sd + mu\n")
                new_src.append(f"{sp}out.loc[s.smiles.values, t] = oof_blended\n")
                new_src.append(f"{sp}print(f\"  stageA {{t:>4}}  n={{len(y):<5}} oof R2={{r2(y, oof_blended):.4f}}  w_lgb={{w_sa[0]:.2f}} w_kr={{w_sa[1]:.2f}}\")\n")
            elif "oof = np.full(len(y), np.nan)" in line:
                new_src.append(line)
                indent = len(line) - len(line.lstrip())
                sp = " " * indent
                new_src.append(f"{sp}oof_p1 = np.full(len(y), np.nan)\n")
                new_src.append(f"{sp}oof_p2 = np.full(len(y), np.nan)\n")
            elif 'print(f"  stageA' in line:
                pass
            else:
                new_src.append(line)
        cell["source"] = new_src
        stageA_done = True

    elif not zoo_done and "def stage_b_models" in src_joined:
        new_source = """def stage_b_models(n):
    small = n < 600
    return {
        "lgb": lambda: lgb.LGBMRegressor(
            n_estimators=120 if FAST else (500 if small else 900),
            learning_rate=0.03, num_leaves=7 if small else 31,
            min_child_samples=5 if small else 20, colsample_bytree=0.7,
            subsample=0.8, subsample_freq=1, reg_lambda=3.0 if small else 1.0,
            verbose=-1, n_jobs=-1, random_state=SEED),
        "ridge": lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30))),
        "et": lambda: ExtraTreesRegressor(
            n_estimators=400, max_features=0.5, min_samples_leaf=2,
            n_jobs=-1, random_state=SEED),
        "xgb": lambda: xgb.XGBRegressor(
            n_estimators=500 if small else 900, learning_rate=0.03,
            max_depth=3 if small else 6, subsample=0.8,
            colsample_bytree=0.5, reg_lambda=2.0,
            tree_method="hist", n_jobs=-1, random_state=SEED,
            verbosity=0),
        "cat": lambda: CatBoostRegressor(
            iterations=500 if small else 900, learning_rate=0.03,
            depth=4 if small else 6, subsample=0.8,
            l2_leaf_reg=3.0, thread_count=-1, random_seed=SEED,
            verbose=0)
    }

train["pred"] = np.nan
test["pred"] = np.nan
rng = np.random.default_rng(SEED)
nf = 3 if FAST else N_FOLDS
report = []

for t in TARGETS:
    mtr = (train.target_type == t).to_numpy()
    sub = train.loc[mtr].reset_index(drop=True)
    y = sub.target.to_numpy(float)
    fold = sub.fold.to_numpy()
    zoo = stage_b_models(len(y))
    mu, sd = y.mean(), y.std() + 1e-9

    oof = {m: np.full(len(y), np.nan) for m in zoo}
    for k in range(nf):
        a, b = np.where(fold != k)[0], np.where(fold == k)[0]
        if not len(b):
            continue
        tbl = fold_tables[k]
        Xs, ys = [], []
        for r in range(1 + (0 if FAST else N_AUG)):
            Xs.append(make_features(sub.smiles.values[a], t, tbl, rng=rng, mask_p=0.0 if r == 0 else 0.35))
            ys.append(y[a])
        Xa, ya = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
        Xb = make_features(sub.smiles.values[b], t, tbl)
        for m, build in zoo.items():
            oof[m][b] = build().fit(Xa, (ya - mu) / sd).predict(Xb) * sd + mu

    P = np.column_stack([np.where(np.isnan(oof[m]), mu, oof[m]) for m in zoo])
    w, _ = sopt.nnls(P, y)
    w = w / w.sum() if w.sum() > 1e-9 else np.full(P.shape[1], 1 / P.shape[1])
    train.loc[mtr, "pred"] = P @ w

    Xs, ys = [], []
    for r in range(1 + (0 if FAST else N_AUG)):
        Xs.append(make_features(sub.smiles.values, t, full_table, rng=rng, mask_p=0.0 if r == 0 else 0.35))
        ys.append(y)
    Xa, ya = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
    mte = (test.target_type == t).to_numpy()
    Xb = make_features(test.loc[mte, "smiles"].values, t, full_table)
    Pt = np.column_stack([build().fit(Xa, (ya - mu) / sd).predict(Xb) * sd + mu
                          for build in zoo.values()])
    test.loc[mte, "pred"] = Pt @ w

    base = r2(y, stageA.loc[sub.smiles.values, t].to_numpy())
    curr = r2(y, P @ w)
    report.append({"target": t, "n": len(y), "stageA_r2": base,
                   "stageB_r2": curr, "gain": curr - base})
    print(f"{t:>4}  StageA {base:.4f} -> StageB {curr:.4f}   ({curr - base:+.4f})   w={{dict(zip(zoo, np.round(w, 2)))}}")

print("\\n", pd.DataFrame(report).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
"""
        cell["source"] = [l + "\n" for l in new_source.split("\n")]
        zoo_done = True

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {OUT}")
