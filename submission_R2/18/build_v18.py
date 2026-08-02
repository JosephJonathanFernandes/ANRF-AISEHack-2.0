"""
build_v18.py — v17 + Stage C (Iterative Sibling Conditioning)
"""
import json, os

IN  = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\17\anrf-v17.ipynb"
OUT = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\18\anrf-v18.ipynb"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(IN, encoding="utf-8") as f:
    nb = json.load(f)

stageB_done = False

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])

    if not stageB_done and "def stage_b_models" in src and "xgb.XGBRegressor" in src:
        # We replace the loop over targets with a double loop (Stage B, Stage C)
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
    }

train["pred"] = np.nan
test["pred"] = np.nan
rng = np.random.default_rng(SEED)
nf = 3 if FAST else N_FOLDS

global _stageA_np, stageA

for stage_name in ["Stage B", "Stage C"]:
    print(f"\\n{'='*40}\\n{stage_name}\\n{'='*40}")
    report = []
    next_stage_mat = np.full((len(uniq), len(TARGETS)), np.nan, dtype=np.float32)

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
                Xs.append(make_features(sub.smiles.values[a], t, tbl,
                                        rng=rng, mask_p=0.0 if r == 0 else 0.35))
                ys.append(y[a])
            Xa, ya = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
            Xb = make_features(sub.smiles.values[b], t, tbl)
            for m, build in zoo.items():
                oof[m][b] = build().fit(Xa, (ya - mu) / sd).predict(Xb) * sd + mu

        P = np.column_stack([np.where(np.isnan(oof[m]), mu, oof[m]) for m in zoo])
        w, _ = sopt.nnls(P, y)
        w = w / w.sum() if w.sum() > 1e-9 else np.full(P.shape[1], 1 / P.shape[1])
        
        pred_oof = P @ w
        train.loc[mtr, "pred"] = pred_oof

        # refit on all training rows and predict all uniq molecules
        Xs, ys = [], []
        for r in range(1 + (0 if FAST else N_AUG)):
            Xs.append(make_features(sub.smiles.values, t, full_table,
                                    rng=rng, mask_p=0.0 if r == 0 else 0.35))
            ys.append(y)
        Xa, ya = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
        
        X_all = make_features(uniq, t, full_table)
        P_all = np.column_stack([build().fit(Xa, (ya - mu) / sd).predict(X_all) * sd + mu
                                 for build in zoo.values()])
        pred_all = P_all @ w
        
        mte = (test.target_type == t).to_numpy()
        test.loc[mte, "pred"] = pred_all[[pos[s] for s in test.loc[mte, "smiles"]]]
        
        t_idx = TARGETS.index(t)
        next_stage_mat[:, t_idx] = pred_all
        sub_idx = [pos[s] for s in sub.smiles.values]
        next_stage_mat[sub_idx, t_idx] = pred_oof

        base = r2(y, stageA.loc[sub.smiles.values, t].to_numpy())
        curr = r2(y, pred_oof)
        report.append({"target": t, "n": len(y), "prev_r2": base,
                       "curr_r2": curr, "gain": curr - base})
        print(f"{t:>4}  Prev {base:.4f} -> Curr {curr:.4f}   "
              f"({curr - base:+.4f})   w={dict(zip(zoo, np.round(w, 2)))}")

    print("\\n", pd.DataFrame(report).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    
    # Update stageA variables for the next iteration
    _stageA_np = next_stage_mat
    stageA = pd.DataFrame(_stageA_np, index=uniq, columns=TARGETS)
"""
        cell["source"] = [l + "\n" for l in new_source.split("\n")]
        stageB_done = True

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {OUT}")
print(f"  stageB_done={stageB_done}")
