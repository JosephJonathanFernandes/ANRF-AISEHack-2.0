"""
AISEHack 2.0 Round 2 — Polymer Property Prediction  (submission_R2 / 7)
=======================================================================
Ensemble of LightGBM, XGBoost, and CatBoost!

Uses the exact best feature subsets found in v6 per-target.
- eps: Set C
- ei: Set B
- eea: Set C
- nc: Set C
- egb: Set D
- egc: Set D
- tg: Set C

Where:
A: desc + Morgan-256
B: desc + Morgan-256 + PI1M
C: desc + Morgan-256 + MACCS
D: desc + Morgan-256 + MACCS + PI1M
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
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "Dataset")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")

TRAIN_PATH  = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH   = os.path.join(DATASET_DIR, "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "submission.csv")

TARGETS = ["egc", "egb", "ei", "eea", "eps", "nc", "tg"]

train      = pd.read_csv(TRAIN_PATH)
test       = pd.read_csv(TEST_PATH)
all_smiles = pd.unique(
    pd.concat([train["smiles"], test["smiles"]], ignore_index=True))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD CACHED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
DESC_LIST = Descriptors.descList
DESC_COLS = [name for name, _ in DESC_LIST]

def _sanitize_df(df):
    """Replace inf, -inf, and |value| > 1e15 with NaN in float columns.
       CRITICAL for XGBoost compatibility."""
    for col in df.select_dtypes(include=["float32", "float64"]).columns:
        s = df[col]
        bad = ~(np.isfinite(s) & (s.abs() < 1e15))
        if bad.any():
            df[col] = s.where(~bad, other=np.nan)
    return df

t0 = time.time()
mol_v3 = pd.read_pickle(os.path.join(CACHE_DIR, "mol_feats_v3.pkl"))
print(f"[cache] mol_feats_v3 {mol_v3.shape}  ({time.time()-t0:.1f}s)")

maccs_cols  = [c for c in mol_v3.columns if c.startswith("maccs_")]
desc_feats   = _sanitize_df(mol_v3[DESC_COLS].copy())
maccs_feats  = mol_v3[maccs_cols].copy()
del mol_v3; gc.collect()

t0 = time.time()
morgan256_feats = pd.read_pickle(os.path.join(CACHE_DIR, "morgan256_v5.pkl"))
print(f"[cache] Morgan-r2-256 {morgan256_feats.shape}  ({time.time()-t0:.1f}s)")

t0 = time.time()
_svd_full = pd.read_pickle(os.path.join(CACHE_DIR, "svd_embeddings_v3.pkl"))
pi1m64 = _svd_full.iloc[:, :64].copy()
pi1m64.columns = [f"svd_{i}" for i in range(64)]
del _svd_full; gc.collect()
print(f"[cache] PI1M-SVD-64 {pi1m64.shape}  ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE CANDIDATES (v6 logic)
# ─────────────────────────────────────────────────────────────────────────────
feats_A = desc_feats.join(morgan256_feats)                          # 473
feats_B = feats_A.join(pi1m64)                                       # 537
feats_C = feats_A.join(maccs_feats)                                  # 640
feats_D = feats_C.join(pi1m64)                                       # 704

BEST_FEATS = {
    "eps": feats_C,
    "ei": feats_B,
    "eea": feats_C,
    "nc": feats_C,
    "egb": feats_D,
    "egc": feats_D,
    "tg": feats_C
}

_HIGH = {"tg", "egc"}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CROSS-PROPERTY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
wide = train.pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="mean")
wide = wide.reindex(columns=TARGETS)
wide.columns = [f"xf_{c}" for c in wide.columns]

def make_X(df, target_type, feats):
    X  = feats.reindex(df["smiles"].values).reset_index(drop=True)
    xf = wide.reindex(df["smiles"].values).reset_index(drop=True).copy()
    xf[f"xf_{target_type}"] = np.nan
    return pd.concat([X, xf], axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENSEMBLE MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
def get_ensemble(tt):
    if tt in _HIGH:
        # High capacity for large datasets
        m_lgb = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.015, num_leaves=127, max_depth=8,
            min_child_samples=20, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=0.5,
            random_state=42, verbosity=-1, n_jobs=4
        )
        m_xgb = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.015, max_depth=7, min_child_weight=15,
            subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=0.5,
            random_state=42, n_jobs=4
        )
        m_cat = CatBoostRegressor(
            iterations=1500, learning_rate=0.02, depth=7, l2_leaf_reg=3,
            subsample=0.8, random_seed=42, verbose=False, thread_count=4
        )
    else:
        # Moderate capacity for small datasets
        m_lgb = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.02, num_leaves=31, max_depth=6,
            min_child_samples=10, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, verbosity=-1, n_jobs=4
        )
        m_xgb = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.02, max_depth=5, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, n_jobs=4
        )
        m_cat = CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=5,
            subsample=0.8, random_seed=42, verbose=False, thread_count=4
        )
    
    return VotingRegressor([
        ('lgb', m_lgb),
        ('xgb', m_xgb),
        ('cat', m_cat)
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CV EVALUATION (ENSEMBLE)
# ─────────────────────────────────────────────────────────────────────────────
def cv_ensemble(target_type, feats, n_splits=5):
    sub    = train[train["target_type"] == target_type].reset_index(drop=True)
    y      = sub["target"].values
    groups = sub["smiles"].values
    X      = make_X(sub, target_type, feats)

    gkf    = GroupKFold(n_splits=n_splits)
    oof    = np.zeros(len(sub))

    for tr_idx, va_idx in gkf.split(X, y, groups):
        ens = get_ensemble(target_type)
        ens.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = ens.predict(X.iloc[va_idx])

    return r2_score(y, oof)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
BASELINE = {"egc":0.9080,"egb":0.7700,"ei":0.8600,"eea":0.8200,"eps":0.8300,"nc":0.7900,"tg":0.9200}
V6_CV    = {"egc":0.8993,"egb":0.8931,"ei":0.8169,"eea":0.8636,"eps":0.7839,"nc":0.8591,"tg":0.9032}

print("\nEvaluating Ensemble CV (LGBM + XGB + CatBoost) ...")
cv_scores = {}
for tt in TARGETS:
    r2 = cv_ensemble(tt, BEST_FEATS[tt])
    cv_scores[tt] = r2
    delta = r2 - V6_CV[tt]
    print(f"  {tt:6s} CV = {r2:7.4f}  (vs v6: {delta:+7.4f})")

mean_new = np.mean(list(cv_scores.values()))
mean_v6  = np.mean(list(V6_CV.values()))
print("-" * 40)
print(f"  MEAN CV = {mean_new:7.4f}  (vs v6: {mean_new - mean_v6:+7.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FINAL FIT & INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\nFitting final ensemble models on full training data ...")
preds = np.zeros(len(test))

for tt in TARGETS:
    mask_tr = (train["target_type"] == tt).values
    mask_te = (test["target_type"]  == tt).values
    feats   = BEST_FEATS[tt]

    Xtr = make_X(train[mask_tr].reset_index(drop=True), tt, feats)
    ytr = train.loc[mask_tr, "target"].values
    Xte = make_X(test[mask_te].reset_index(drop=True), tt, feats)

    ens = get_ensemble(tt)
    ens.fit(Xtr, ytr)
    preds[mask_te] = ens.predict(Xte)
    print(f"  {tt}: Done. Mean Pred = {preds[mask_te].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SAVE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "target": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved  →  {OUTPUT_PATH}")
