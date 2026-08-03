"""
AISEHack 2.0 Round 2 — Polymer Property Prediction  (submission_R2 / 5)
=======================================================================
FULL ROOT-CAUSE DIAGNOSIS (from v3/v4 failures):

  PROBLEM 1 — Wrong fingerprint type:
    Morgan-r3-2048 (1.7%% density) is too sparse for small-data targets
    (eps n=229, ei n=222). The baseline Morgan-r2-256 (8.6%% density)
    gives better per-bit information for small-sample regression.

  PROBLEM 2 — SVD compression loses structural resolution:
    LightGBM handles sparse binary bits efficiently (it is literally
    designed for this). Compressing 4263 bits → 128 SVD dims removes
    structural distinctions that LGB could exploit natively.

  PROBLEM 3 — Ridge blend hurts small-data targets:
    Ridge R² was 0.65-0.75 for eps/ei, but LGB was 0.76-0.81.
    A 20%% Ridge weight dragged the blend BELOW the LGB alone.

  SOLUTION (submission_R2/5):
    ✓ Revert Morgan FP → radius-2, 256-bit (same as baseline that scored 0.855)
    ✓ ADD MACCS Keys 167-bit from v3 cache (free, domain-specific)
    ✓ Per-target feature selection: test (desc+M256+MACCS) vs
      (desc+M256+MACCS+PI1M-SVD-64). Pick winner per target like v1 did.
    ✓ LightGBM-only (no Ridge, no XGBoost), slightly better params:
        num_leaves 15→31 (low-data) / 63 (high-data), lr 0.03→0.02
    ✓ Early stopping on the held-out GroupKFold val fold
    ✓ Proper inf/huge-value sanitization (< 1e15 clip)

  Expected: recover eps/ei to ~baseline, keep gains on egb/eea/nc/egc/tg.
  Target: mean R² > 0.865.

Caches reused from submission_R2/3: mol_feats_v3.pkl, svd_embeddings_v3.pkl
New cache: morgan256_v5.pkl  (~30 sec to compute for 10605 SMILES)
"""

import warnings
warnings.filterwarnings("ignore")

import os, gc, time
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys

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
# 1.  LOAD CACHED mol_feats_v3  (has desc + Morgan-2048 + MACCS + RDKit-FP)
#     We extract desc and MACCS from it; Morgan-2048 is discarded (too sparse)
# ─────────────────────────────────────────────────────────────────────────────
DESC_LIST = Descriptors.descList
DESC_COLS = [name for name, _ in DESC_LIST]

MOL_V3_CACHE = os.path.join(CACHE_DIR, "mol_feats_v3.pkl")
t0 = time.time()
mol_v3 = pd.read_pickle(MOL_V3_CACHE)
print(f"[cache] mol_feats_v3 {mol_v3.shape}  ({time.time()-t0:.1f}s)")

# Extract: (a) desc columns, (b) MACCS columns
maccs_cols = [c for c in mol_v3.columns if c.startswith("maccs_")]
desc_feats  = mol_v3[DESC_COLS].copy()
maccs_feats = mol_v3[maccs_cols].copy()
del mol_v3; gc.collect()

# Sanitize desc: clip inf and huge-finite values (e.g. Ipc descriptor)
for col in desc_feats.columns:
    s = desc_feats[col]
    bad = ~(np.isfinite(s) & (s.abs() < 1e15))
    if bad.any():
        desc_feats[col] = s.where(~bad, other=np.nan)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MORGAN-r2-256 FINGERPRINTS — recompute (SAME as original baseline)
#     These have 8.6%% density vs 1.7%% for the r3-2048 used in v3/v4.
#     Better signal-to-noise for small-data targets.
# ─────────────────────────────────────────────────────────────────────────────
M256_CACHE = os.path.join(CACHE_DIR, "morgan256_v5.pkl")

if os.path.exists(M256_CACHE):
    t0 = time.time()
    morgan256_feats = pd.read_pickle(M256_CACHE)
    print(f"[cache] Morgan-r2-256 {morgan256_feats.shape}  ({time.time()-t0:.1f}s)")
else:
    t0 = time.time()
    print(f"Computing Morgan-r2-256 for {len(all_smiles)} SMILES ...")
    rows = []
    for idx, smi in enumerate(all_smiles):
        if idx % 500 == 0:
            print(f"  {idx}/{len(all_smiles)}", end="\r", flush=True)
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(256, dtype=np.uint8)
        if mol is not None:
            for b in AllChem.GetMorganFingerprintAsBitVect(
                    mol, 2, nBits=256).GetOnBits():
                arr[b] = 1
        rows.append(arr)
    print()
    morgan256_feats = pd.DataFrame(
        np.vstack(rows), index=all_smiles,
        columns=[f"m256_{i}" for i in range(256)])
    morgan256_feats.to_pickle(M256_CACHE)
    print(f"Saved Morgan-r2-256 {morgan256_feats.shape}  ({time.time()-t0:.0f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PI1M SVD EMBEDDINGS — truncate to 64 dims (from v3 cache of 256 dims)
#     Used selectively per target (only if it improves CV)
# ─────────────────────────────────────────────────────────────────────────────
SVD_CACHE = os.path.join(CACHE_DIR, "svd_embeddings_v3.pkl")
t0 = time.time()
_full_svd = pd.read_pickle(SVD_CACHE)
pi1m_svd64 = _full_svd.iloc[:, :64].copy()
pi1m_svd64.columns = [f"svd_{i}" for i in range(64)]
del _full_svd; gc.collect()
print(f"[cache] PI1M-SVD-64 {pi1m_svd64.shape}  ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  BUILD FEATURE MATRICES
#     feats_base: desc(217) + Morgan-r2-256(256) + MACCS(167)  = 640 features
#     feats_plus: feats_base + PI1M-SVD-64(64)                  = 704 features
# ─────────────────────────────────────────────────────────────────────────────
feats_base = desc_feats.join(morgan256_feats).join(maccs_feats)
feats_plus = feats_base.join(pi1m_svd64)
print(f"feats_base: {feats_base.shape}  feats_plus: {feats_plus.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CROSS-PROPERTY FEATURES (same as all previous versions)
# ─────────────────────────────────────────────────────────────────────────────
wide = train.pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="mean")
wide = wide.reindex(columns=TARGETS)
wide.columns = [f"xf_{c}" for c in wide.columns]


def make_X(df, target_type, feats):
    """Feature matrix with cross-property features, own-target masked."""
    X = feats.reindex(df["smiles"].values).reset_index(drop=True)
    xf = wide.reindex(df["smiles"].values).reset_index(drop=True).copy()
    xf[f"xf_{target_type}"] = np.nan   # never see own target
    return pd.concat([X, xf], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LGB HYPERPARAMETERS — per-target size-based tuning
#     Low-data (<500 samples): conservative (fewer leaves, more regularisation)
#     High-data (>=500 samples): deeper trees for more capacity
#     All use early stopping on the val fold.
# ─────────────────────────────────────────────────────────────────────────────
_LOW_DATA  = {"nc", "eps", "ei", "eea", "egb"}  # n < 400
_HIGH_DATA = {"tg", "egc"}                       # n > 2000

def get_lgb_params(tt):
    base = dict(random_state=42, verbosity=-1, n_jobs=2,
                subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_alpha=0.5)
    if tt in _LOW_DATA:
        # Careful: n~200-350 per target; GroupKFold train fold ~160-280 samples
        base.update(n_estimators=800, learning_rate=0.02,
                    num_leaves=31, max_depth=5,
                    min_child_samples=10, reg_lambda=2.0)
    else:
        # n=2028 (egc) or 4143 (tg) — allow deeper, faster trees
        base.update(n_estimators=1500, learning_rate=0.02,
                    num_leaves=63, max_depth=7,
                    min_child_samples=20, reg_lambda=1.0)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 7.  5-FOLD GROUPED CROSS-VALIDATION
#     Tests feats_base vs feats_plus per target, picks the winner.
# ─────────────────────────────────────────────────────────────────────────────
def cv_lgb(target_type, feats, n_splits=5):
    """OOF R² for LightGBM with GroupKFold on SMILES — zero leakage."""
    sub    = train[train["target_type"] == target_type].reset_index(drop=True)
    y      = sub["target"].values
    groups = sub["smiles"].values
    X      = make_X(sub, target_type, feats)

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(sub))
    params = get_lgb_params(target_type)

    for tr_idx, va_idx in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=[(X.iloc[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(60, verbose=False),
                          lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X.iloc[va_idx])

    return r2_score(y, oof)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  RUN CV WITH OLD-VS-NEW COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
BASELINE = {"egc": 0.9080, "egb": 0.7700, "ei": 0.8600, "eea": 0.8200,
            "eps": 0.8300, "nc": 0.7900, "tg": 0.9200}

print()
hdr = (f"{'Target':6s}  {'N':>5s}  {'Baseline':>9s}  "
       f"{'Base feats':>10s}  {'+PI1M-64':>9s}  {'Chosen':>8s}  {'Delta':>7s}")
print(hdr)
print("-" * len(hdr))

chosen_feats = {}
cv_scores    = {}

for tt in TARGETS:
    n       = int((train["target_type"] == tt).sum())
    r_base  = cv_lgb(tt, feats_base)
    r_plus  = cv_lgb(tt, feats_plus)
    winner  = "plus" if r_plus > r_base else "base"
    best_r2 = max(r_base, r_plus)
    chosen_feats[tt] = feats_plus if winner == "plus" else feats_base
    cv_scores[tt]    = best_r2
    delta = best_r2 - BASELINE.get(tt, 0.0)
    print(f"{tt:6s}  {n:5d}  {BASELINE.get(tt,0):9.4f}  "
          f"{r_base:10.4f}  {r_plus:9.4f}  {winner:>8s}  {delta:+7.4f}")

mean_base  = np.mean(list(BASELINE.values()))
mean_new   = np.mean(list(cv_scores.values()))
delta_mean = mean_new - mean_base
print("-" * len(hdr))
print(f"{'MEAN':6s}  {'':5s}  {mean_base:9.4f}  {'':10s}  {'':9s}  "
      f"{'':8s}  {delta_mean:+7.4f}   →  {mean_new:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FINAL FIT ON FULL TRAINING DATA → PREDICT TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\nFitting final models on full training data ...")
preds = np.zeros(len(test))

for tt in TARGETS:
    mask_tr = (train["target_type"] == tt).values
    mask_te = (test["target_type"]  == tt).values
    feats   = chosen_feats[tt]

    Xtr = make_X(train[mask_tr].reset_index(drop=True), tt, feats)
    ytr = train.loc[mask_tr, "target"].values
    Xte = make_X(test[mask_te].reset_index(drop=True), tt, feats)

    m = lgb.LGBMRegressor(**get_lgb_params(tt))
    m.fit(Xtr, ytr)
    preds[mask_te] = m.predict(Xte)
    print(f"  {tt}: mean_pred={preds[mask_te].mean():.3f}  "
          f"feats={'base+PI1M' if feats is feats_plus else 'base'}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  SAVE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "target": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved  →  {OUTPUT_PATH}")
print(f"  Shape:  {submission.shape}")
print(submission.head(10).to_string(index=False))
print(f"\n  Mean CV R2       : {mean_new:.4f}")
print(f"  Delta vs baseline: {delta_mean:+.4f}")
