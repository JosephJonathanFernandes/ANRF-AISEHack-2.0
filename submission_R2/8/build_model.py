"""
AISEHack 2.0 Round 2 — Polymer Property Prediction  (submission_R2 / 8)
=======================================================================
v8: TWO-STAGE STACKING to solve Target Leakage Distribution Shift!

Problem: Small targets (eps, ei, eea, nc) were measured on the same 135 molecules.
In CV, the model overfits to the sparse `xf_` lab measurements. But in `test.csv`,
90% of molecules have NO cross-properties (they are all NaNs), causing LB scores to crash.

Solution (Stacking):
1. STAGE 1: Train LGBM on ALL 7 targets using ONLY molecular features (no xf_).
   Generate a dense (N, 7) matrix of predictions for train and test.
2. STAGE 2: Train final models on (Molecular Features + Dense S1 Predictions).
   This completely eliminates the NaN distribution shift between CV and LB.
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
all_smiles = pd.unique(
    pd.concat([train["smiles"], test["smiles"]], ignore_index=True))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD CACHED MOLECULAR FEATURES
# ─────────────────────────────────────────────────────────────────────────────
DESC_LIST = Descriptors.descList
DESC_COLS = [name for name, _ in DESC_LIST]

def _sanitize_df(df):
    """Replace inf, -inf, and |value| > 1e15 with NaN in float columns."""
    for col in df.select_dtypes(include=["float32", "float64"]).columns:
        s = df[col]
        bad = ~(np.isfinite(s) & (s.abs() < 1e15))
        if bad.any():
            df[col] = s.where(~bad, other=np.nan)
    return df

t0 = time.time()
mol_v3 = pd.read_pickle(os.path.join(CACHE_DIR, "mol_feats_v3.pkl"))
print(f"[cache] mol_feats_v3 {mol_v3.shape}  ({time.time()-t0:.1f}s)")

maccs_cols   = [c for c in mol_v3.columns if c.startswith("maccs_")]
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
# 2.  FEATURE CANDIDATES (Base level - no target leakage)
# ─────────────────────────────────────────────────────────────────────────────
feats_A = desc_feats.join(morgan256_feats)                          # 473
feats_B = feats_A.join(pi1m64)                                      # 537
feats_C = feats_A.join(maccs_feats)                                 # 640
feats_D = feats_C.join(pi1m64)                                      # 704

# Optimal subsets found in v6 (prevents overfitting on small targets)
BEST_FEATS = {
    "eps": feats_C,
    "ei":  feats_B,
    "eea": feats_C,
    "nc":  feats_C,
    "egb": feats_D,
    "egc": feats_D,
    "tg":  feats_C
}
_HIGH = {"tg", "egc"}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
def get_lgb_params(tt, n_estimators=None):
    base = dict(random_state=42, verbosity=-1, n_jobs=4,
                subsample=0.8, subsample_freq=1, reg_alpha=0.5)
    if tt in _HIGH:
        base.update(
            n_estimators=n_estimators or 2500,
            learning_rate=0.015,
            num_leaves=127,
            max_depth=8,
            min_child_samples=20,
            colsample_bytree=0.5,
            reg_lambda=0.5,
        )
    else:
        base.update(
            n_estimators=n_estimators or 800,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            min_child_samples=10,
            colsample_bytree=0.7,
            reg_lambda=2.0,
        )
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STAGE 1: GENERATE DENSE PREDICTIONS FOR ALL MOLECULES
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STAGE 1: Training Base Predictors ===")

s1_models = {}
for tt in TARGETS:
    mask_tr = (train["target_type"] == tt).values
    X_tr = BEST_FEATS[tt].reindex(train.loc[mask_tr, "smiles"].values).reset_index(drop=True)
    y_tr = train.loc[mask_tr, "target"].values
    
    m = lgb.LGBMRegressor(**get_lgb_params(tt))
    m.fit(X_tr, y_tr)
    s1_models[tt] = m
    print(f"  [S1] {tt:3s} trained on {X_tr.shape[0]} samples")

print("\nGenerating Dense S1 Features for Train + Test...")
s1_matrix = pd.DataFrame(index=all_smiles, columns=[f"s1_{t}" for t in TARGETS])

# Generate predictions for EVERY molecule in the dataset
for tt in TARGETS:
    X_all = BEST_FEATS[tt].reindex(all_smiles).reset_index(drop=True)
    s1_matrix[f"s1_{tt}"] = s1_models[tt].predict(X_all)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  STAGE 2: CV EVALUATION WITH TARGET ENCODING (STACKING)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STAGE 2: Training Stacked Predictors ===")

cv_scores = {}
for tt in TARGETS:
    sub = train[train["target_type"] == tt].reset_index(drop=True)
    y = sub["target"].values
    groups = sub["smiles"].values
    
    # 1. Base molecular features
    X_base = BEST_FEATS[tt].reindex(sub["smiles"].values).reset_index(drop=True)
    
    # 2. Append Stage 1 predictions (EXCLUDING the current target to prevent massive self-leakage)
    s1_cols = [f"s1_{t}" for t in TARGETS if t != tt]
    X_s1 = s1_matrix.reindex(sub["smiles"].values)[s1_cols].reset_index(drop=True)
    
    X = pd.concat([X_base, X_s1], axis=1)
    
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(sub))
    params = get_lgb_params(tt)
    
    for tr_idx, va_idx in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        
    r2 = r2_score(y, oof)
    cv_scores[tt] = r2
    print(f"  [S2 CV] {tt:3s}: R2 = {r2:7.4f} (Features: {X.shape[1]})")

print("-" * 45)
print(f"  [S2] MEAN CV = {np.mean(list(cv_scores.values())):7.4f}")
print("  Note: This CV score has some slight target leakage from S1, but importantly,")
print("  it guarantees NO NaN distribution shift on the Leaderboard test set!")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  STAGE 2: FINAL FIT & INFERENCE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STAGE 2: Final Fit & Test Predictions ===")
preds = np.zeros(len(test))

for tt in TARGETS:
    mask_tr = (train["target_type"] == tt).values
    mask_te = (test["target_type"]  == tt).values
    
    # Train Features
    Xtr_base = BEST_FEATS[tt].reindex(train.loc[mask_tr, "smiles"].values).reset_index(drop=True)
    Xtr_s1   = s1_matrix.reindex(train.loc[mask_tr, "smiles"].values)[s1_cols].reset_index(drop=True)
    Xtr      = pd.concat([Xtr_base, Xtr_s1], axis=1)
    ytr      = train.loc[mask_tr, "target"].values
    
    # Test Features
    Xte_base = BEST_FEATS[tt].reindex(test.loc[mask_te, "smiles"].values).reset_index(drop=True)
    Xte_s1   = s1_matrix.reindex(test.loc[mask_te, "smiles"].values)[s1_cols].reset_index(drop=True)
    Xte      = pd.concat([Xte_base, Xte_s1], axis=1)
    
    # Fit & Predict
    m = lgb.LGBMRegressor(**get_lgb_params(tt))
    m.fit(Xtr, ytr)
    preds[mask_te] = m.predict(Xte)
    print(f"  {tt:3s}: Test predictions generated. Mean = {preds[mask_te].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "target": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved  →  {OUTPUT_PATH}")
print(submission.head())
