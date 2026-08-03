"""
AISEHack 2.0 Round 2 — Polymer Property Prediction  (submission_R2 / 6)
=======================================================================
DIAGNOSIS of v5 remaining regressions (LB 0.862 vs target >0.875):

  v5 feats_base = desc + Morgan-r2-256 + MACCS(167)  ← ALWAYS included MACCS
  v5 feats_plus = feats_base + PI1M-SVD-64

  Problem: never tested WITHOUT MACCS for small-data targets!
  MACCS adds 167 extra features on top of n~183 training samples per fold.
  For eps (n=229) and ei (n=222), MACCS caused -0.045 regression each.

FIX in submission_R2/6:
  1. Four feature-set candidates per target — systematic per-target search:
       A: desc + Morgan-256                          (473)  no MACCS, no PI1M
       B: desc + Morgan-256 + PI1M-64               (537)  no MACCS, with PI1M
       C: desc + Morgan-256 + MACCS                 (640)  with MACCS, no PI1M
       D: desc + Morgan-256 + MACCS + PI1M-64       (704)  with MACCS, with PI1M
     For high-data (tg, egc) also:
       E: desc + Morgan-2048 + MACCS + PI1M-64      (2496) large FP for big datasets
  2. Record avg best_iteration from CV early-stopping, use it for final fit
     (removes bias: CV stops early due to smaller train set, but final uses
     the converged iteration count)
  3. Higher LGB capacity for high-data: num_leaves=127 (was 63)
  4. Lower lr for high-data: 0.015 (was 0.02) with more trees: 3000 (was 1500)

Expected: recover eps/ei/tg/egc to baseline levels while keeping egb/nc/eea gains.
Target: mean CV R2 > 0.875 → LB > 0.870.

All caches reused (mol_feats_v3, morgan256_v5, svd_embeddings_v3).
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
# 1.  LOAD CACHED FEATURES
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


# ── mol_feats_v3: has desc + Morgan-2048 + MACCS + RDKit-FP
t0 = time.time()
mol_v3 = pd.read_pickle(os.path.join(CACHE_DIR, "mol_feats_v3.pkl"))
print(f"[cache] mol_feats_v3 {mol_v3.shape}  ({time.time()-t0:.1f}s)")

maccs_cols  = [c for c in mol_v3.columns if c.startswith("maccs_")]
mfp2048_cols= [c for c in mol_v3.columns if c.startswith("mfp_")]

desc_feats   = _sanitize_df(mol_v3[DESC_COLS].copy())
maccs_feats  = mol_v3[maccs_cols].copy()
mfp2048_feats= mol_v3[mfp2048_cols].copy()
del mol_v3; gc.collect()

# ── Morgan-r2-256 (computed in v5, cached)
t0 = time.time()
morgan256_feats = pd.read_pickle(
    os.path.join(CACHE_DIR, "morgan256_v5.pkl"))
print(f"[cache] Morgan-r2-256 {morgan256_feats.shape}  ({time.time()-t0:.1f}s)")

# ── PI1M SVD-64 (from v3 cache, truncated)
t0 = time.time()
_svd_full = pd.read_pickle(os.path.join(CACHE_DIR, "svd_embeddings_v3.pkl"))
pi1m64 = _svd_full.iloc[:, :64].copy()
pi1m64.columns = [f"svd_{i}" for i in range(64)]
del _svd_full; gc.collect()
print(f"[cache] PI1M-SVD-64 {pi1m64.shape}  ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE CANDIDATES
#     A: desc + Morgan-256                 (473) — baseline-style, no MACCS
#     B: desc + Morgan-256 + PI1M-64       (537)
#     C: desc + Morgan-256 + MACCS         (640)
#     D: desc + Morgan-256 + MACCS + PI1M  (704) — v5 "plus"
#     E: desc + Morgan-2048 + MACCS + PI1M (2496) — only for high-data targets
# ─────────────────────────────────────────────────────────────────────────────
feats_A = desc_feats.join(morgan256_feats)                          # 473
feats_B = feats_A.join(pi1m64)                                       # 537
feats_C = feats_A.join(maccs_feats)                                  # 640
feats_D = feats_C.join(pi1m64)                                       # 704
feats_E = desc_feats.join(mfp2048_feats).join(maccs_feats).join(pi1m64)  # 2496

CANDIDATES_LOW  = {"A": feats_A, "B": feats_B, "C": feats_C, "D": feats_D}
CANDIDATES_HIGH = {"A": feats_A, "C": feats_C, "D": feats_D, "E": feats_E}
_HIGH = {"tg", "egc"}

print(f"\nFeature sets built: A={feats_A.shape[1]} B={feats_B.shape[1]} "
      f"C={feats_C.shape[1]} D={feats_D.shape[1]} E={feats_E.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CROSS-PROPERTY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
wide = train.pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="mean")
wide = wide.reindex(columns=TARGETS)
wide.columns = [f"xf_{c}" for c in wide.columns]


def make_X(df, target_type, feats):
    """Feature matrix with xf_ features; own-target always masked."""
    X  = feats.reindex(df["smiles"].values).reset_index(drop=True)
    xf = wide.reindex(df["smiles"].values).reset_index(drop=True).copy()
    xf[f"xf_{target_type}"] = np.nan
    return pd.concat([X, xf], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PER-TARGET LGB HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
def get_lgb_params(tt, n_estimators=None):
    base = dict(random_state=42, verbosity=-1, n_jobs=2,
                subsample=0.8, subsample_freq=1, reg_alpha=0.5)
    if tt in _HIGH:
        base.update(
            n_estimators=n_estimators or 3000,
            learning_rate=0.015,
            num_leaves=127,
            max_depth=8,
            min_child_samples=20,
            colsample_bytree=0.5,    # sample 50% of ~2500 features per tree
            reg_lambda=0.5,
        )
    else:
        base.update(
            n_estimators=n_estimators or 1000,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            min_child_samples=10,
            colsample_bytree=0.7,
            reg_lambda=2.0,
        )
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GROUPED CV — returns (oof_r2, avg_best_iteration)
#     Records the avg early-stopping iteration across folds.
#     This is used as n_estimators for the final fit (no early-stopping bias).
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
              callbacks=[lgb.early_stopping(80, verbose=False),
                          lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        iters.append(m.best_iteration_ if m.best_iteration_ > 0
                      else params["n_estimators"])

    avg_iter = max(int(np.mean(iters) * 1.10), 50)  # add 10% buffer
    return r2_score(y, oof), avg_iter


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PER-TARGET FEATURE SELECTION + CV TABLE
# ─────────────────────────────────────────────────────────────────────────────
V5 = {"egc":0.8993,"egb":0.8927,"ei":0.8143,"eea":0.8628,
      "eps":0.7851,"nc":0.8589,"tg":0.9016}
BASELINE = {"egc":0.9080,"egb":0.7700,"ei":0.8600,"eea":0.8200,
            "eps":0.8300,"nc":0.7900,"tg":0.9200}

print()
hdr = (f"{'Target':6s}  {'N':>5s}  {'Baseline':>9s}  {'v5':>7s}  "
       + "".join(f"  {k:>7s}" for k in ["A","B","C","D","E"])
       + f"  {'Best':>6s}  {'Iter':>5s}  {'Delta':>7s}")
print(hdr); print("-" * len(hdr))

chosen_feats  = {}
best_iters    = {}
cv_scores     = {}

for tt in TARGETS:
    n          = int((train["target_type"] == tt).sum())
    candidates = CANDIDATES_HIGH if tt in _HIGH else CANDIDATES_LOW

    results = {}
    for label, fset in candidates.items():
        r2, avg_it = cv_lgb(tt, fset)
        results[label] = (r2, avg_it)

    best_label = max(results, key=lambda k: results[k][0])
    best_r2, best_it = results[best_label]
    chosen_feats[tt] = candidates[best_label]
    best_iters[tt]   = best_it
    cv_scores[tt]    = best_r2

    scores_str = "".join(
        f"  {results[k][0]:7.4f}" if k in results else "         "
        for k in ["A","B","C","D","E"])
    delta = best_r2 - BASELINE[tt]
    print(f"{tt:6s}  {n:5d}  {BASELINE[tt]:9.4f}  {V5[tt]:7.4f}"
          + scores_str
          + f"  {best_label:>6s}  {best_it:5d}  {delta:+7.4f}")

mean_base  = np.mean(list(BASELINE.values()))
mean_v5    = np.mean(list(V5.values()))
mean_new   = np.mean(list(cv_scores.values()))
print("-" * len(hdr))
print(f"{'MEAN':6s}  {'':5s}  {mean_base:9.4f}  {mean_v5:7.4f}"
      + "  " * 5 + "         "
      + f"  {'':6s}  {'':5s}  {mean_new - mean_base:+7.4f}  → {mean_new:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FINAL FIT — use avg_best_iter from CV (no early-stopping bias)
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

    # Final model: converged to avg CV iteration (10% buffer already applied)
    params = get_lgb_params(tt, n_estimators=n_iter)
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr)   # no early stopping: use exactly n_iter trees
    preds[mask_te] = m.predict(Xte)
    print(f"  {tt}: feats={chosen_feats[tt].shape[1]}  "
          f"n_trees={n_iter}  mean_pred={preds[mask_te].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SAVE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "target": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved  →  {OUTPUT_PATH}")
print(f"  Shape:  {submission.shape}")
print(submission.head(10).to_string(index=False))
print(f"\n  Mean CV R2       : {mean_new:.4f}")
print(f"  Delta vs baseline: {mean_new - mean_base:+.4f}")
print(f"  Delta vs v5 CV   : {mean_new - mean_v5:+.4f}")
