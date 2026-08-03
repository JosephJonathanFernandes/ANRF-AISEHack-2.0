"""
AISEHack 2.0 Round 2 — Polymer Property Prediction  (submission_R2 / 9)
=======================================================================
v9: Reverting to v6 Architecture + Advanced Fingerprints (Avalon, TT, AP)

Diagnosis of v8: Stacking replaced 100% accurate (but sparse) lab measurements 
with dense but noisy S1 predictions. This caused the LB score to drop from 0.862 to 0.854.
The test set has 10% overlap with train for cross-features, and for those 10%, 
the perfectly accurate lab measurements are crucial!

In v9, we revert to the v6 feature setup (keeping xf_ features as they are), 
but we introduce 3072 new features:
1. Avalon (1024)
2. Topological Torsion (1024)
3. Atom Pair (1024)

We will run a systematic per-target feature selection across these new fingerprints
to break the 0.862 ceiling without relying on Gensim Word2Vec.
"""

import warnings
warnings.filterwarnings("ignore")

import os, gc, time
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "Dataset")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_PATH  = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH   = os.path.join(DATASET_DIR, "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "submission.csv")

TARGETS = ["egc", "egb", "ei", "eea", "eps", "nc", "tg"]

train      = pd.read_csv(TRAIN_PATH)
test       = pd.read_csv(TEST_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD CACHED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
DESC_LIST = Descriptors.descList
DESC_COLS = [name for name, _ in DESC_LIST]

def _sanitize_df(df):
    for col in df.select_dtypes(include=["float32", "float64"]).columns:
        s = df[col]
        bad = ~(np.isfinite(s) & (s.abs() < 1e15))
        if bad.any():
            df[col] = s.where(~bad, other=np.nan)
    return df

t0 = time.time()
mol_v3 = pd.read_pickle(os.path.join(BASE_DIR, "..", "6", "cache", "mol_feats_v3.pkl"))
maccs_cols  = [c for c in mol_v3.columns if c.startswith("maccs_")]
desc_feats  = _sanitize_df(mol_v3[DESC_COLS].copy())
maccs_feats = mol_v3[maccs_cols].copy()
del mol_v3; gc.collect()

morgan256_feats = pd.read_pickle(os.path.join(BASE_DIR, "..", "6", "cache", "morgan256_v5.pkl"))

_svd_full = pd.read_pickle(os.path.join(BASE_DIR, "..", "6", "cache", "svd_embeddings_v3.pkl"))
pi1m64 = _svd_full.iloc[:, :64].copy()
pi1m64.columns = [f"svd_{i}" for i in range(64)]
del _svd_full; gc.collect()

ap = pd.read_pickle(os.path.join(CACHE_DIR, "ap_1024.pkl"))
tt = pd.read_pickle(os.path.join(CACHE_DIR, "tt_1024.pkl"))
av = pd.read_pickle(os.path.join(CACHE_DIR, "av_1024.pkl"))

print(f"Features loaded in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────
feats_Base = desc_feats.join(morgan256_feats)                                # 473
feats_v6_best = feats_Base.join(maccs_feats).join(pi1m64)                    # 704
feats_AP = feats_v6_best.join(ap)                                            # 1728
feats_TT = feats_v6_best.join(tt)                                            # 1728
feats_AV = feats_v6_best.join(av)                                            # 1728
feats_ALL = feats_v6_best.join(ap).join(tt).join(av)                         # 3776

CANDIDATES = {
    "Base": feats_Base,
    "v6": feats_v6_best,
    "AP": feats_AP,
    "TT": feats_TT,
    "AV": feats_AV,
    "ALL": feats_ALL
}
_HIGH = {"tg", "egc"}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CROSS-PROPERTY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
wide = train.pivot_table(index="smiles", columns="target_type", values="target", aggfunc="mean")
wide = wide.reindex(columns=TARGETS)
wide.columns = [f"xf_{c}" for c in wide.columns]

def make_X(df, target_type, feats):
    X  = feats.reindex(df["smiles"].values).reset_index(drop=True)
    xf = wide.reindex(df["smiles"].values).reset_index(drop=True).copy()
    xf[f"xf_{target_type}"] = np.nan
    return pd.concat([X, xf], axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PER-TARGET LGB HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
def get_lgb_params(tt, n_estimators=None):
    base = dict(random_state=42, verbosity=-1, n_jobs=4,
                subsample=0.8, subsample_freq=1, reg_alpha=0.5)
    if tt in _HIGH:
        base.update(
            n_estimators=n_estimators or 3000,
            learning_rate=0.015,
            num_leaves=127,
            max_depth=8,
            min_child_samples=20,
            colsample_bytree=0.2, # lowered because of many features
            reg_lambda=0.5,
        )
    else:
        base.update(
            n_estimators=n_estimators or 1000,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            min_child_samples=10,
            colsample_bytree=0.3, # lowered because of many features
            reg_lambda=2.0,
        )
    return base

# ─────────────────────────────────────────────────────────────────────────────
# 5.  GROUPED CV
# ─────────────────────────────────────────────────────────────────────────────
def cv_lgb(target_type, feats, n_splits=5):
    sub    = train[train["target_type"] == target_type].reset_index(drop=True)
    y      = sub["target"].values
    groups = sub["smiles"].values
    X      = make_X(sub, target_type, feats)

    gkf    = GroupKFold(n_splits=n_splits)
    oof    = np.zeros(len(sub))
    params = get_lgb_params(target_type)
    iters  = []

    for tr_idx, va_idx in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=[(X.iloc[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        iters.append(m.best_iteration_ if m.best_iteration_ > 0 else params["n_estimators"])

    avg_iter = max(int(np.mean(iters) * 1.10), 50)
    return r2_score(y, oof), avg_iter


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEATURE SELECTION & TRAINING
# ─────────────────────────────────────────────────────────────────────────────
V6 = {"egc":0.9003,"egb":0.8931,"ei":0.8169,"eea":0.8636,
      "eps":0.7839,"nc":0.8591,"tg":0.9032}

print(f"{'Target':6s}  {'v6_CV':>7s}  {'Base':>7s}  {'v6':>7s}  {'AP':>7s}  {'TT':>7s}  {'AV':>7s}  {'ALL':>7s}  {'Best':>6s}  {'Iter':>5s}  {'Delta':>7s}")

chosen_feats  = {}
best_iters    = {}
cv_scores     = {}

for tt in TARGETS:
    results = {}
    for label, fset in CANDIDATES.items():
        r2, avg_it = cv_lgb(tt, fset)
        results[label] = (r2, avg_it)

    best_label = max(results, key=lambda k: results[k][0])
    best_r2, best_it = results[best_label]
    chosen_feats[tt] = CANDIDATES[best_label]
    best_iters[tt]   = best_it
    cv_scores[tt]    = best_r2

    scores_str = "".join(f"  {results[k][0]:7.4f}" for k in ["Base", "v6", "AP", "TT", "AV", "ALL"])
    delta = best_r2 - V6[tt]
    print(f"{tt:6s}  {V6[tt]:7.4f}{scores_str}  {best_label:>6s}  {best_it:5d}  {delta:+7.4f}")

mean_v6  = np.mean(list(V6.values()))
mean_new = np.mean(list(cv_scores.values()))
print(f"{'MEAN':6s}  {mean_v6:7.4f}" + "  " * 6 + "         " + f"  {'':6s}  {'':5s}  {mean_new - mean_v6:+7.4f}  → {mean_new:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FINAL FIT
# ─────────────────────────────────────────────────────────────────────────────
print("\nFitting final models on full training data ...")
preds = np.zeros(len(test))

for tt in TARGETS:
    mask_tr = (train["target_type"] == tt).values
    mask_te = (test["target_type"]  == tt).values
    feats   = chosen_feats[tt]
    n_iter  = best_iters[tt]

    Xtr = make_X(train[mask_tr].reset_index(drop=True), tt, feats)
    ytr = train.loc[mask_tr, "target"].values
    Xte = make_X(test[mask_te].reset_index(drop=True), tt, feats)

    params = get_lgb_params(tt, n_estimators=n_iter)
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr)
    preds[mask_te] = m.predict(Xte)

submission = pd.DataFrame({"id": test["id"], "target": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved Output to {OUTPUT_PATH}")
