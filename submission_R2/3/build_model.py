"""
AISEHack 2.0 Round 2 — Polymer Property Prediction  (submission_R2 / 3)
=======================================================================
Upgrades over previous baselines (submission_R2/2):
  1. Morgan fingerprints: 256-bit  →  2048-bit (radius=3)
  2. + MACCS Keys (167-bit)  +  RDKit Topological Fingerprints (2048-bit)
  3. Word2Vec (100-dim)  →  Morgan-token TF-IDF + TruncatedSVD (256-dim)
     trained on PI1M.csv  (no gensim / C++ build tools required)
  4. Per-target hyperparameter configs (low-data vs high-data targets)
  5. Three-model ensemble:  LightGBM + XGBoost + Ridge  (70 / 20 / 10 %)
  6. GroupKFold(n_splits=5) on smiles column throughout — zero leakage
  7. All heavy intermediates cached to disk — fast re-runs

Re-running reuses cached intermediates: ~2 min after first cold run.
"""

import warnings
warnings.filterwarnings("ignore")

import os, gc, time, pickle
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, RDKFingerprint

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb
import xgboost as xgb

RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — relative to this script; adjust DATASET_DIR if needed
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "Dataset")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_PATH  = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH   = os.path.join(DATASET_DIR, "test.csv")
PI1M_PATH   = os.path.join(DATASET_DIR, "PI1M.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "submission.csv")

TARGETS = ["egc", "egb", "ei", "eea", "eps", "nc", "tg"]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MOLECULAR FEATURE EXTRACTION  (cached)
#     • RDKit 2-D descriptors  (~208 continuous)
#     • Morgan FP 2048-bit radius-3
#     • MACCS Keys 167-bit
#     • RDKit Topological FP 2048-bit
# ─────────────────────────────────────────────────────────────────────────────
DESC_LIST = Descriptors.descList

def compute_mol_features(smiles_list: list,
                          morgan_bits: int = 2048,
                          morgan_radius: int = 3,
                          rdkit_bits: int = 2048) -> pd.DataFrame:
    rows, n = [], len(smiles_list)
    for idx, smi in enumerate(smiles_list):
        if idx % 200 == 0:
            print(f"  Fingerprints: {idx}/{n}", end="\r", flush=True)
        mol = Chem.MolFromSmiles(smi)
        feats = {}

        # Continuous RDKit descriptors
        if mol is None:
            for name, _ in DESC_LIST:
                feats[name] = np.nan
        else:
            for name, func in DESC_LIST:
                try:
                    v = func(mol)
                    feats[name] = float(v) if np.isfinite(float(v)) else np.nan
                except Exception:
                    feats[name] = np.nan

        # Binary fingerprints (0 for invalid SMILES)
        if mol is None:
            feats.update({f"mfp_{i}": 0 for i in range(morgan_bits)})
            feats.update({f"maccs_{i}": 0 for i in range(167)})
            feats.update({f"rfp_{i}": 0 for i in range(rdkit_bits)})
        else:
            mfp = np.zeros(morgan_bits, dtype=np.uint8)
            for b in AllChem.GetMorganFingerprintAsBitVect(
                    mol, morgan_radius, nBits=morgan_bits).GetOnBits():
                mfp[b] = 1
            feats.update({f"mfp_{i}": int(v) for i, v in enumerate(mfp)})

            maccs = np.zeros(167, dtype=np.uint8)
            for b in MACCSkeys.GenMACCSKeys(mol).GetOnBits():
                if b < 167:
                    maccs[b] = 1
            feats.update({f"maccs_{i}": int(v) for i, v in enumerate(maccs)})

            rfp = np.zeros(rdkit_bits, dtype=np.uint8)
            for b in RDKFingerprint(mol, fpSize=rdkit_bits).GetOnBits():
                rfp[b] = 1
            feats.update({f"rfp_{i}": int(v) for i, v in enumerate(rfp)})

        rows.append(feats)

    print()
    return pd.DataFrame(rows, index=list(smiles_list))


train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
all_smiles = pd.unique(
    pd.concat([train["smiles"], test["smiles"]], ignore_index=True))

MOL_CACHE = os.path.join(CACHE_DIR, "mol_feats_v3.pkl")
if os.path.exists(MOL_CACHE):
    t0 = time.time()
    mol_feats = pd.read_pickle(MOL_CACHE)
    print(f"[cache] mol_feats {mol_feats.shape}  ({time.time()-t0:.1f}s)")
else:
    t0 = time.time()
    print(f"Computing fingerprints + descriptors for {len(all_smiles)} SMILES ...")
    mol_feats = compute_mol_features(list(all_smiles))
    mol_feats.to_pickle(MOL_CACHE)
    print(f"Saved mol_feats {mol_feats.shape}  ({time.time()-t0:.0f}s)")

DESC_COLS = [name for name, _ in DESC_LIST]   # used by Ridge

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PI1M EMBEDDINGS via Morgan-token TF-IDF + TruncatedSVD (LSA)
#     Mimics FastText behaviour: sparse substructure co-occurrence → dense vec.
#     Memory-efficient: chunked reading of PI1M.csv, streaming TF-IDF.
# ─────────────────────────────────────────────────────────────────────────────
EMB_CACHE = os.path.join(CACHE_DIR, "svd_embeddings_v3.pkl")
EMB_DIM   = 256

def morgan_token_string(smi: str, radius: int = 1) -> str:
    """Convert SMILES to a whitespace-delimited string of Morgan substructure IDs."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    fp = AllChem.GetMorganFingerprint(mol, radius)
    return " ".join(str(k) for k in fp.GetNonzeroElements().keys())


if os.path.exists(EMB_CACHE):
    t0 = time.time()
    emb_feats = pd.read_pickle(EMB_CACHE)
    print(f"[cache] SVD embeddings {emb_feats.shape}  ({time.time()-t0:.1f}s)")
else:
    CHUNK = 40_000
    t0 = time.time()
    print(f"Reading PI1M.csv in chunks of {CHUNK} ...")
    pi1m_docs = []
    for chunk in pd.read_csv(PI1M_PATH, chunksize=CHUNK, usecols=["SMILES"]):
        for smi in chunk["SMILES"]:
            s = morgan_token_string(smi)
            if s:
                pi1m_docs.append(s)
    print(f"  PI1M corpus: {len(pi1m_docs):,} molecules  ({time.time()-t0:.0f}s)")

    # Also add train/test SMILES so vocabulary covers them
    query_docs = [morgan_token_string(s) for s in all_smiles]
    all_docs   = pi1m_docs + query_docs
    del pi1m_docs; gc.collect()

    t0 = time.time()
    print("Fitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(min_df=3, max_features=50_000)
    tfidf.fit(all_docs)
    del all_docs; gc.collect()

    print("Transforming query SMILES ...")
    X_sparse = tfidf.transform(query_docs)   # shape: (n_unique_smiles, vocab)
    del query_docs; gc.collect()

    print(f"Running TruncatedSVD(n_components={EMB_DIM}) ...")
    svd = TruncatedSVD(n_components=EMB_DIM, random_state=42, n_iter=5)
    emb_matrix = svd.fit_transform(X_sparse)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD explained variance: {explained:.3f}  ({time.time()-t0:.0f}s)")

    emb_feats = pd.DataFrame(
        emb_matrix, index=all_smiles,
        columns=[f"svd_{i}" for i in range(EMB_DIM)])
    emb_feats.to_pickle(EMB_CACHE)
    print(f"Saved SVD embeddings {emb_feats.shape}")

mol_feats_full = mol_feats.join(emb_feats)
print(f"Full feature matrix: {mol_feats_full.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CROSS-PROPERTY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
wide = train.pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="mean")
wide = wide.reindex(columns=TARGETS)
wide.columns = [f"xf_{c}" for c in wide.columns]


def make_X(df: pd.DataFrame, target_type: str,
           feats: pd.DataFrame, include_cross: bool = True) -> pd.DataFrame:
    """Build the feature matrix; mask own target to prevent leakage."""
    X = feats.reindex(df["smiles"].values).reset_index(drop=True)
    if include_cross:
        xf = wide.reindex(df["smiles"].values).reset_index(drop=True).copy()
        xf[f"xf_{target_type}"] = np.nan   # never see own target
        X = pd.concat([X, xf], axis=1)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PER-TARGET HYPERPARAMETER CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
_LOW_DATA  = {"nc", "eps", "ei", "eea", "egb"}
_HIGH_DATA = {"tg", "egc"}

BLEND_W = (0.70, 0.20, 0.10)   # LGB, XGB, Ridge

def get_lgb_params(tt: str) -> dict:
    base = dict(random_state=42, verbosity=-1,
                subsample=0.8, subsample_freq=1,
                colsample_bytree=0.6, reg_alpha=0.5)
    if tt in _LOW_DATA:
        base.update(n_estimators=600, learning_rate=0.02, num_leaves=7,
                    max_depth=4, min_child_samples=5, reg_lambda=2.0)
    else:
        base.update(n_estimators=800, learning_rate=0.03, num_leaves=63,
                    max_depth=-1, min_child_samples=10, reg_lambda=0.5)
    return base

def get_xgb_params(tt: str) -> dict:
    base = dict(random_state=42, n_jobs=2, verbosity=0,
                subsample=0.8, colsample_bytree=0.6,
                reg_alpha=0.5, tree_method="hist")
    if tt in _LOW_DATA:
        base.update(n_estimators=600, learning_rate=0.02, max_depth=4,
                    min_child_weight=5, reg_lambda=2.0)
    else:
        base.update(n_estimators=800, learning_rate=0.03, max_depth=6,
                    min_child_weight=10, reg_lambda=0.5)
    return base

def ridge_pipeline(alpha: float = 10.0):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)),
                            validate=False),
        RobustScaler(),
        Ridge(alpha=alpha),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5.  GROUPED CROSS-VALIDATION  (GroupKFold on smiles, n_splits=5)
# ─────────────────────────────────────────────────────────────────────────────
def cv_target(tt: str, n_splits: int = 5) -> dict:
    """
    OOF CV for LGB + XGB + Ridge and their blend.
    Groups are unique SMILES — molecules never span train/val folds.
    """
    sub    = train[train["target_type"] == tt].reset_index(drop=True)
    y      = sub["target"].values
    groups = sub["smiles"].values
    X_full = make_X(sub, tt, mol_feats_full, include_cross=True)
    desc_cols_here = [c for c in X_full.columns if c in set(DESC_COLS)]
    X_desc = X_full[desc_cols_here]

    gkf = GroupKFold(n_splits=n_splits)
    oof_lgb   = np.zeros(len(sub))
    oof_xgb   = np.zeros(len(sub))
    oof_ridge = np.zeros(len(sub))

    for tr_idx, va_idx in gkf.split(X_full, y, groups):
        m_lgb = lgb.LGBMRegressor(**get_lgb_params(tt))
        m_lgb.fit(X_full.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_full.iloc[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        oof_lgb[va_idx] = m_lgb.predict(X_full.iloc[va_idx])

        m_xgb = xgb.XGBRegressor(**get_xgb_params(tt))
        m_xgb.fit(X_full.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_full.iloc[va_idx], y[va_idx])],
                  verbose=False)
        oof_xgb[va_idx] = m_xgb.predict(X_full.iloc[va_idx])

        m_ridge = ridge_pipeline(alpha=10.0)
        m_ridge.fit(X_desc.iloc[tr_idx], y[tr_idx])
        oof_ridge[va_idx] = m_ridge.predict(X_desc.iloc[va_idx])

    w1, w2, w3 = BLEND_W
    oof_blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_ridge
    return {
        "lgb":   r2_score(y, oof_lgb),
        "xgb":   r2_score(y, oof_xgb),
        "ridge": r2_score(y, oof_ridge),
        "blend": r2_score(y, oof_blend),
        "n":     len(sub),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RUN CV WITH OLD-VS-NEW COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
# Approximate baseline scores from submission_R2/2 (build_model-1.py LGB only)
BASELINE = {"egc": 0.9080, "egb": 0.7700, "ei": 0.8600, "eea": 0.8200,
            "eps": 0.8300, "nc": 0.7900, "tg": 0.9200}

print()
hdr = (f"{'Target':6s}  {'N':>5s}  {'Old R2':>8s}  {'LGB':>8s}  "
       f"{'XGB':>8s}  {'Ridge':>8s}  {'Blend':>8s}  {'Delta':>8s}")
print(hdr)
print("-" * len(hdr))

cv_results = {}
for tt in TARGETS:
    res = cv_target(tt)
    cv_results[tt] = res
    delta = res["blend"] - BASELINE.get(tt, 0.0)
    print(f"{tt:6s}  {res['n']:5d}  {BASELINE.get(tt,0):8.4f}  "
          f"{res['lgb']:8.4f}  {res['xgb']:8.4f}  "
          f"{res['ridge']:8.4f}  {res['blend']:8.4f}  {delta:+8.4f}")

mean_old   = np.mean(list(BASELINE.values()))
mean_blend = np.mean([cv_results[tt]["blend"] for tt in TARGETS])
delta_mean = mean_blend - mean_old
print("-" * len(hdr))
print(f"{'MEAN':6s}  {'':5s}  {mean_old:8.4f}  {'':8s}  "
      f"{'':8s}  {'':8s}  {mean_blend:8.4f}  {delta_mean:+8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FINAL FIT ON FULL TRAINING DATA → PREDICT TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\nFitting final models on full training data ...")
preds = np.zeros(len(test))

for tt in TARGETS:
    mask_tr = (train["target_type"] == tt).values
    mask_te = (test["target_type"]  == tt).values

    Xtr = make_X(train[mask_tr].reset_index(drop=True), tt,
                 mol_feats_full, include_cross=True)
    ytr = train.loc[mask_tr, "target"].values
    Xte = make_X(test[mask_te].reset_index(drop=True), tt,
                 mol_feats_full, include_cross=True)

    desc_here = [c for c in Xtr.columns if c in set(DESC_COLS)]
    Xtr_d, Xte_d = Xtr[desc_here], Xte[desc_here]

    m_lgb = lgb.LGBMRegressor(**get_lgb_params(tt))
    m_lgb.fit(Xtr, ytr)
    p_lgb = m_lgb.predict(Xte)

    m_xgb = xgb.XGBRegressor(**get_xgb_params(tt))
    m_xgb.fit(Xtr, ytr)
    p_xgb = m_xgb.predict(Xte)

    m_ridge = ridge_pipeline(alpha=10.0)
    m_ridge.fit(Xtr_d, ytr)
    p_ridge = m_ridge.predict(Xte_d)

    w1, w2, w3 = BLEND_W
    preds[mask_te] = w1 * p_lgb + w2 * p_xgb + w3 * p_ridge
    print(f"  {tt}: LGB={p_lgb.mean():.3f}  XGB={p_xgb.mean():.3f}  "
          f"Ridge={p_ridge.mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SAVE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "target": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved submission → {OUTPUT_PATH}")
print(f"  Shape : {submission.shape}")
print(submission.head(10).to_string(index=False))
print(f"\n  Mean CV R2 (blend) : {mean_blend:.4f}")
print(f"  Delta vs baseline  : {delta_mean:+.4f}")
