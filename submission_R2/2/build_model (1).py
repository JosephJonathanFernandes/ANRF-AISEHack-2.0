"""
AISEHack 2.0 Round 2 - Polymer Property Prediction
Improved baseline: RDKit descriptors + Morgan fingerprints + cross-property features
"""
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
import lightgbm as lgb

RDLogger.DisableLog('rdApp.*')

TARGETS = ['egc', 'egb', 'ei', 'eea', 'eps', 'nc', 'tg']

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
test = pd.read_csv('/mnt/user-data/uploads/test.csv')

# ---------------------------------------------------------------
# 1. Molecular features (RDKit descriptors + Morgan fingerprint)
# ---------------------------------------------------------------
DESC_LIST = Descriptors.descList

def compute_mol_features(smiles_list, n_bits=256, radius=2):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        feats = {}
        if mol is None:
            for name, _ in DESC_LIST:
                feats[name] = np.nan
            for i in range(n_bits):
                feats[f'fp_{i}'] = 0
        else:
            for name, func in DESC_LIST:
                try:
                    v = func(mol)
                    feats[name] = v if np.isfinite(v) else np.nan
                except Exception:
                    feats[name] = np.nan
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=int)
            for b in fp.GetOnBits():
                arr[b] = 1
            for i in range(n_bits):
                feats[f'fp_{i}'] = arr[i]
        rows.append(feats)
    return pd.DataFrame(rows, index=list(smiles_list))

CACHE_PATH = '/home/claude/mol_feats_cache.pkl'
import os
all_smiles = pd.unique(pd.concat([train['smiles'], test['smiles']], ignore_index=True))
t0 = time.time()
if os.path.exists(CACHE_PATH):
    mol_feats = pd.read_pickle(CACHE_PATH)
    print(f'Loaded cached feature matrix: {mol_feats.shape}  ({time.time()-t0:.1f}s)')
else:
    print(f'Computing molecular features for {len(all_smiles)} unique SMILES...')
    mol_feats = compute_mol_features(all_smiles)
    mol_feats.to_pickle(CACHE_PATH)
    print(f'Feature matrix shape: {mol_feats.shape}  ({time.time()-t0:.1f}s)')

# ---------------------------------------------------------------
# 2. Cross-property features: known OTHER target values for same molecule
# ---------------------------------------------------------------
wide = train.pivot_table(index='smiles', columns='target_type', values='target', aggfunc='mean')
wide = wide.reindex(columns=TARGETS)
wide.columns = [f'xf_{c}' for c in wide.columns]

def make_X(df, target_type, use_cross=True):
    X = mol_feats.reindex(df['smiles'].values).reset_index(drop=True)
    if use_cross:
        xf = wide.reindex(df['smiles'].values).reset_index(drop=True).copy()
        xf[f'xf_{target_type}'] = np.nan  # never see own target as a feature
        X = pd.concat([X, xf], axis=1)
    return X

LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15,
    min_child_samples=10, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=0.5,
    random_state=42, verbosity=-1
)

def cv_score(target_type, use_cross):
    sub = train[train['target_type'] == target_type].reset_index(drop=True)
    y = sub['target'].values
    groups = sub['smiles'].values
    X = make_X(sub, target_type, use_cross)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(sub))
    for tr_idx, va_idx in gkf.split(X, y, groups):
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X.iloc[va_idx])
    return r2_score(y, oof)

DESC_COLS = [name for name, _ in DESC_LIST]

def cv_score_ridge_baseline(target_type):
    """Descriptor-only Ridge, mirroring the competition's baseline_model.ipynb approach."""
    sub = train[train['target_type'] == target_type].reset_index(drop=True)
    y = sub['target'].values
    groups = sub['smiles'].values
    X = mol_feats.reindex(sub['smiles'].values)[DESC_COLS].reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(sub))
    for tr_idx, va_idx in gkf.split(X, y, groups):
        model = make_pipeline(
            SimpleImputer(strategy='median'),
            FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a))),  # tame huge-range descriptors like Ipc
            RobustScaler(),
            Ridge(alpha=10.0),
        )
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X.iloc[va_idx])
    return r2_score(y, oof)

print(f'\n{"target":6s} {"n":>5s} {"R2 Ridge":>10s} {"R2 LGBM":>10s} {"R2 +cross":>12s} {"delta":>8s}')
results = {}
for tt in TARGETS:
    n = int((train['target_type'] == tt).sum())
    r2_ridge = cv_score_ridge_baseline(tt)
    r2_base = cv_score(tt, use_cross=False)
    r2_cross = cv_score(tt, use_cross=True)
    results[tt] = (r2_ridge, r2_base, r2_cross)
    print(f'{tt:6s} {n:5d} {r2_ridge:10.4f} {r2_base:10.4f} {r2_cross:12.4f} {r2_cross-r2_base:8.4f}')

mean_ridge = np.mean([v[0] for v in results.values()])
mean_base = np.mean([v[1] for v in results.values()])
mean_cross = np.mean([v[2] for v in results.values()])
print(f'{"MEAN":6s} {"":5s} {mean_ridge:10.4f} {mean_base:10.4f} {mean_cross:12.4f} {mean_cross-mean_base:8.4f}')

# ---------------------------------------------------------------
# 3. Final fit on full train, predict test
# ---------------------------------------------------------------
preds = np.zeros(len(test))
for tt in TARGETS:
    mask_tr = (train['target_type'] == tt).values
    Xtr = make_X(train[mask_tr].reset_index(drop=True), tt, use_cross=True)
    ytr = train.loc[mask_tr, 'target'].values
    mask_te = (test['target_type'] == tt).values
    Xte = make_X(test[mask_te].reset_index(drop=True), tt, use_cross=True)
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(Xtr, ytr)
    preds[mask_te] = model.predict(Xte)

submission = pd.DataFrame({'id': test['id'], 'target': preds})
submission.to_csv('/mnt/user-data/outputs/submission.csv', index=False)
print('\nSaved submission:', submission.shape)
print(submission.head())
