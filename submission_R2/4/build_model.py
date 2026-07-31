"""
AISEHack 2.0 Round 2 — Polymer Property Prediction  (submission_R2 / 4)
=======================================================================
Root-cause fix over submission_R2/3 (which scored 0.848 < 0.855 baseline):

  PROBLEM: 4263 sparse binary FP bits (85% zeros) at 4736 total features
  caused LightGBM to dilute feature-importance budget across mostly-zero
  columns, hurting especially the small-data targets (eps, ei, tg).

  SOLUTION: Compress [Morgan-2048 + MACCS-167 + RDKitFP-2048] → SVD(128)
  dense fingerprint representation.  This:
    • Reduces FP dimension 4263 → 128 (-97%)
    • Creates dense orthogonal "structural theme" features
    • Total features: 217 desc + 128 fp_svd + 128 pi1m_svd + 7 xf = 480
    • Models can now split efficiently without drowning in sparse bits

  Other changes vs v3:
    • Drop XGBoost (NaN instability + marginal gain) → LGB 80% + Ridge 20%
    • Better per-target LGB regularisation for small-data targets
    • Ridge alpha 100 for low-data / 10 for high-data
    • All heavy caches REUSED from submission_R2/3 (no cold-start needed)

Expected runtime: ~30 min (all descriptors + PI1M SVD cached, FP-SVD new ~1 min)
"""

import warnings
warnings.filterwarnings("ignore")

import os, gc, time
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
PI1M_PATH   = os.path.join(DATASET_DIR, "PI1M.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "submission.csv")

TARGETS = ["egc", "egb", "ei", "eea", "eps", "nc", "tg"]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD / COMPUTE MOLECULAR FEATURES  (reuse v3 cache)
# ─────────────────────────────────────────────────────────────────────────────
DESC_LIST = Descriptors.descList
DESC_COLS = [name for name, _ in DESC_LIST]   # used by Ridge

def compute_mol_features(smiles_list, morgan_bits=2048, morgan_radius=3,
                          rdkit_bits=2048):
    """Compute RDKit descriptors + Morgan + MACCS + RDKit-FP for each SMILES."""
    rows, n = [], len(smiles_list)
    for idx, smi in enumerate(smiles_list):
        if idx % 200 == 0:
            print(f"  Fingerprints: {idx}/{n}", end="\r", flush=True)
        mol = Chem.MolFromSmiles(smi)
        feats = {}
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


train      = pd.read_csv(TRAIN_PATH)
test       = pd.read_csv(TEST_PATH)
all_smiles = pd.unique(
    pd.concat([train["smiles"], test["smiles"]], ignore_index=True))

MOL_CACHE = os.path.join(CACHE_DIR, "mol_feats_v3.pkl")
if os.path.exists(MOL_CACHE):
    t0 = time.time()
    mol_feats = pd.read_pickle(MOL_CACHE)
    print(f"[cache] mol_feats {mol_feats.shape}  ({time.time()-t0:.1f}s)")
else:
    t0 = time.time()
    print(f"Computing mol features for {len(all_smiles)} SMILES ...")
    mol_feats = compute_mol_features(list(all_smiles))
    mol_feats.to_pickle(MOL_CACHE)
    print(f"Saved mol_feats {mol_feats.shape}  ({time.time()-t0:.0f}s)")

# Global inf / huge-value sanitization on float columns
_float_cols = mol_feats.select_dtypes(include=["float32", "float64"]).columns
for _col in _float_cols:
    _s = mol_feats[_col]
    mol_feats[_col] = _s.where(np.isfinite(_s) & (_s.abs() < 1e15), other=np.nan)
del _float_cols

# ─────────────────────────────────────────────────────────────────────────────
# 2.  COMPRESS FINGERPRINT BITS → SVD(128)  [cached]
#
#  The 4263 sparse binary bits (85% zeros) dilute LightGBM feature budget.
#  SVD compresses them into 128 dense orthogonal "structural theme" vectors,
#  similar to how LSA works for text.  This is the #1 fix vs submission_R2/3.
# ─────────────────────────────────────────────────────────────────────────────
FP_SVD_CACHE = os.path.join(CACHE_DIR, "fp_svd128_v4.pkl")
FP_SVD_DIM   = 128

fp_bit_cols = [c for c in mol_feats.columns
               if c.startswith(("mfp_", "maccs_", "rfp_"))]

if os.path.exists(FP_SVD_CACHE):
    t0 = time.time()
    fp_svd_feats = pd.read_pickle(FP_SVD_CACHE)
    print(f"[cache] FP-SVD {fp_svd_feats.shape}  ({time.time()-t0:.1f}s)")
else:
    t0 = time.time()
    print(f"Running TruncatedSVD({FP_SVD_DIM}) on {len(fp_bit_cols)} FP bits ...")
    fp_mat = mol_feats[fp_bit_cols].values.astype(np.float32)
    svd_fp = TruncatedSVD(n_components=FP_SVD_DIM, random_state=42, n_iter=7)
    fp_svd_mat = svd_fp.fit_transform(fp_mat)
    expl = svd_fp.explained_variance_ratio_.sum()
    print(f"  FP-SVD explained variance: {expl:.3f}  ({time.time()-t0:.0f}s)")
    fp_svd_feats = pd.DataFrame(
        fp_svd_mat, index=mol_feats.index,
        columns=[f"fpsvd_{i}" for i in range(FP_SVD_DIM)])
    fp_svd_feats.to_pickle(FP_SVD_CACHE)
    print(f"Saved FP-SVD {fp_svd_feats.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PI1M EMBEDDINGS — TF-IDF + SVD(128)  [reuse v3 cache]
# ─────────────────────────────────────────────────────────────────────────────
EMB_CACHE = os.path.join(CACHE_DIR, "svd_embeddings_v3.pkl")
EMB_DIM   = 128   # we'll truncate the 256-dim cache to first 128 dims

def morgan_token_string(smi, radius=1):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    fp = AllChem.GetMorganFingerprint(mol, radius)
    return " ".join(str(k) for k in fp.GetNonzeroElements().keys())

if os.path.exists(EMB_CACHE):
    t0 = time.time()
    _emb_full = pd.read_pickle(EMB_CACHE)
    # Take only first 128 dims to reduce feature count vs v3's 256
    emb_feats = _emb_full.iloc[:, :EMB_DIM].copy()
    emb_feats.columns = [f"svd_{i}" for i in range(EMB_DIM)]
    del _emb_full
    print(f"[cache] PI1M-SVD {emb_feats.shape}  ({time.time()-t0:.1f}s)")
else:
    CHUNK = 40_000
    t0 = time.time()
    print(f"Reading PI1M.csv in chunks ...")
    pi1m_docs = []
    for chunk in pd.read_csv(PI1M_PATH, chunksize=CHUNK, usecols=["SMILES"]):
        for smi in chunk["SMILES"]:
            s = morgan_token_string(smi)
            if s:
                pi1m_docs.append(s)
    query_docs = [morgan_token_string(s) for s in all_smiles]
    all_docs   = pi1m_docs + query_docs
    del pi1m_docs; gc.collect()
    tfidf = TfidfVectorizer(min_df=3, max_features=50_000)
    tfidf.fit(all_docs)
    del all_docs; gc.collect()
    X_sparse = tfidf.transform(query_docs)
    del query_docs; gc.collect()
    svd = TruncatedSVD(n_components=256, random_state=42, n_iter=5)
    emb_mat = svd.fit_transform(X_sparse)
    full_emb = pd.DataFrame(emb_mat, index=all_smiles,
                            columns=[f"svd_{i}" for i in range(256)])
    full_emb.to_pickle(EMB_CACHE)
    emb_feats = full_emb.iloc[:, :EMB_DIM].copy()
    print(f"Saved PI1M-SVD, using first {EMB_DIM} dims")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ASSEMBLE FINAL FEATURE MATRIX
#     desc (217) + fp_svd (128) + pi1m_svd (128) + xf (7) = 480 features
#     → no more 4736-dim sparse mess
# ─────────────────────────────────────────────────────────────────────────────
desc_only  = mol_feats[[c for c in mol_feats.columns if c not in fp_bit_cols]]
mol_feats_full = desc_only.join(fp_svd_feats).join(emb_feats)
print(f"Full feature matrix: {mol_feats_full.shape}  "
      f"(desc={len([c for c in mol_feats_full.columns if c in set(DESC_COLS)])} "
      f"fpsvd={FP_SVD_DIM} pi1msvd={EMB_DIM})")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CROSS-PROPERTY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
wide = train.pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="mean")
wide = wide.reindex(columns=TARGETS)
wide.columns = [f"xf_{c}" for c in wide.columns]


def make_X(df, target_type, feats, include_cross=True):
    """Build feature matrix; mask own-target xf_ col; sanitize huge floats."""
    X = feats.reindex(df["smiles"].values).reset_index(drop=True)
    if include_cross:
        xf = wide.reindex(df["smiles"].values).reset_index(drop=True).copy()
        xf[f"xf_{target_type}"] = np.nan
        X = pd.concat([X, xf], axis=1)
    # Catch any remaining inf or enormous finite values (safety net)
    float_cols = X.select_dtypes(include=["float32", "float64"]).columns
    for col in float_cols:
        s = X[col]
        bad = ~(np.isfinite(s) & (s.abs() < 1e15))
        if bad.any():
            X[col] = s.where(~bad, other=np.nan)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PER-TARGET HYPERPARAMETER CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
# Train-set sizes: egb=337, eps=229, nc=229, ei=222, eea=221  (low-data)
#                  egc=2028, tg=4143                           (high-data)
_LOW_DATA  = {"nc", "eps", "ei", "eea", "egb"}
_HIGH_DATA = {"tg", "egc"}

BLEND_W = (0.80, 0.20)   # LGB, Ridge

def get_lgb_params(tt):
    base = dict(random_state=42, verbosity=-1, n_jobs=2,
                subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_alpha=0.5,
                importance_type="gain")
    if tt in _LOW_DATA:
        # Heavy regularisation for tiny datasets (<350 samples)
        base.update(n_estimators=500, learning_rate=0.01,
                    num_leaves=15, max_depth=5,
                    min_child_samples=10, reg_lambda=5.0)
    else:
        # Allow deeper trees for large datasets
        base.update(n_estimators=1000, learning_rate=0.02,
                    num_leaves=63, max_depth=6,
                    min_child_samples=20, reg_lambda=1.0)
    return base

def ridge_pipeline(alpha=10.0):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)),
                            validate=False),
        RobustScaler(),
        Ridge(alpha=alpha),
    )

def get_ridge_alpha(tt):
    return 100.0 if tt in _LOW_DATA else 10.0

# ─────────────────────────────────────────────────────────────────────────────
# 7.  CROSS-VALIDATION  (GroupKFold on smiles)
# ─────────────────────────────────────────────────────────────────────────────
def cv_target(tt, n_splits=5):
    """OOF CV for LGB + Ridge blend.  Groups = unique SMILES (zero leakage)."""
    sub    = train[train["target_type"] == tt].reset_index(drop=True)
    y      = sub["target"].values
    groups = sub["smiles"].values
    X_full = make_X(sub, tt, mol_feats_full, include_cross=True)
    # Ridge uses only the continuous RDKit descriptors
    desc_here = [c for c in X_full.columns if c in set(DESC_COLS)]
    X_desc = X_full[desc_here]

    gkf = GroupKFold(n_splits=n_splits)
    oof_lgb   = np.zeros(len(sub))
    oof_ridge = np.zeros(len(sub))

    lgb_p = get_lgb_params(tt)

    for tr_idx, va_idx in gkf.split(X_full, y, groups):
        # LightGBM with early stopping on the validation fold
        m_lgb = lgb.LGBMRegressor(**lgb_p)
        m_lgb.fit(X_full.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_full.iloc[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        oof_lgb[va_idx] = m_lgb.predict(X_full.iloc[va_idx])

        # Ridge on descriptor-only features
        m_ridge = ridge_pipeline(alpha=get_ridge_alpha(tt))
        m_ridge.fit(X_desc.iloc[tr_idx], y[tr_idx])
        oof_ridge[va_idx] = m_ridge.predict(X_desc.iloc[va_idx])

    w1, w2 = BLEND_W
    oof_blend = w1 * oof_lgb + w2 * oof_ridge
    return {
        "lgb":   r2_score(y, oof_lgb),
        "ridge": r2_score(y, oof_ridge),
        "blend": r2_score(y, oof_blend),
        "n":     len(sub),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8.  RUN CV WITH COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
# Approximate baseline from submission_R2/2 for reference
BASELINE = {"egc": 0.9080, "egb": 0.7700, "ei": 0.8600, "eea": 0.8200,
            "eps": 0.8300, "nc": 0.7900, "tg": 0.9200}
# Previous v3 results (scored 0.848 on LB)
V3       = {"egc": 0.9000, "egb": 0.8927, "ei": 0.8134, "eea": 0.8647,
            "eps": 0.7658, "nc": 0.8497, "tg": 0.0}  # tg crashed in v3

print()
hdr = (f"{'Target':6s}  {'N':>5s}  {'Baseline':>9s}  {'v3 blend':>9s}  "
       f"{'v4 LGB':>8s}  {'v4 Ridge':>9s}  {'v4 Blend':>9s}  {'Delta':>8s}")
print(hdr)
print("-" * len(hdr))

cv_results = {}
for tt in TARGETS:
    res = cv_target(tt)
    cv_results[tt] = res
    delta = res["blend"] - BASELINE.get(tt, 0.0)
    print(f"{tt:6s}  {res['n']:5d}  {BASELINE.get(tt,0):9.4f}  "
          f"{V3.get(tt,0):9.4f}  {res['lgb']:8.4f}  "
          f"{res['ridge']:9.4f}  {res['blend']:9.4f}  {delta:+8.4f}")

mean_base  = np.mean(list(BASELINE.values()))
mean_v3    = np.mean([V3[tt] for tt in TARGETS])
mean_blend = np.mean([cv_results[tt]["blend"] for tt in TARGETS])
delta_mean = mean_blend - mean_base
print("-" * len(hdr))
print(f"{'MEAN':6s}  {'':5s}  {mean_base:9.4f}  {mean_v3:9.4f}  "
      f"{'':8s}  {'':9s}  {mean_blend:9.4f}  {delta_mean:+8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FINAL FIT ON FULL TRAINING DATA → PREDICT TEST
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

    m_lgb = lgb.LGBMRegressor(**get_lgb_params(tt))
    m_lgb.fit(Xtr, ytr)
    p_lgb = m_lgb.predict(Xte)

    m_ridge = ridge_pipeline(alpha=get_ridge_alpha(tt))
    m_ridge.fit(Xtr[desc_here], ytr)
    p_ridge = m_ridge.predict(Xte[desc_here])

    w1, w2 = BLEND_W
    preds[mask_te] = w1 * p_lgb + w2 * p_ridge
    print(f"  {tt}: LGB={p_lgb.mean():.3f}  Ridge={p_ridge.mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "target": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved  →  {OUTPUT_PATH}")
print(f"  Shape:  {submission.shape}")
print(submission.head(10).to_string(index=False))
print(f"\n  Mean CV R2 (blend) : {mean_blend:.4f}")
print(f"  Delta vs baseline  : {delta_mean:+.4f}")
