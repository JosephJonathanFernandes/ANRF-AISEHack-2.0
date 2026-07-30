"""
AISEHack 2.0 Round 2 - Polymer Property Prediction
Pipeline: RDKit descriptors + Morgan fingerprints + cross-property features
          + PI1M word2vec embeddings (trained from scratch, kept only where
          it improves CV per target). No pretrained models or external data,
          per Round 2 rules.

Re-running this script reuses cached intermediates (descriptors, PI1M corpus,
word2vec model) so it takes ~1 min instead of ~8 min after the first run.
"""
import warnings
warnings.filterwarnings('ignore')
import os
import time
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from gensim.models import Word2Vec
import lightgbm as lgb

RDLogger.DisableLog('rdApp.*')
CACHE_DIR = '/home/claude'
TARGETS = ['egc', 'egb', 'ei', 'eea', 'eps', 'nc', 'tg']

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
test = pd.read_csv('/mnt/user-data/uploads/test.csv')
all_smiles = pd.unique(pd.concat([train['smiles'], test['smiles']], ignore_index=True))

# ---------------------------------------------------------------
# 1. RDKit descriptors + Morgan fingerprint (cached)
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

desc_cache = f'{CACHE_DIR}/mol_feats_cache.pkl'
if os.path.exists(desc_cache):
    mol_feats = pd.read_pickle(desc_cache)
else:
    t0 = time.time()
    mol_feats = compute_mol_features(all_smiles)
    mol_feats.to_pickle(desc_cache)
    print(f'Computed descriptors+FP: {mol_feats.shape} ({time.time()-t0:.0f}s)')

# ---------------------------------------------------------------
# 2. PI1M word2vec embeddings, trained from scratch (cached)
# ---------------------------------------------------------------
def morgan_sentence(smi, radius=1):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprint(mol, radius)
    return [str(k) for k in fp.GetNonzeroElements().keys()]

w2v_cache = f'{CACHE_DIR}/w2v_embeddings.pkl'
if os.path.exists(w2v_cache):
    w2v_feats = pd.read_pickle(w2v_cache)
else:
    t0 = time.time()
    pi1m = pd.read_csv('/mnt/user-data/uploads/PI1M.csv')
    sentences = [s for s in (morgan_sentence(x) for x in pi1m['SMILES']) if s]
    print(f'Built PI1M corpus: {len(sentences)} sentences ({time.time()-t0:.0f}s)')

    t0 = time.time()
    w2v = Word2Vec(sentences=sentences, vector_size=100, window=10, min_count=3,
                    sg=0, workers=1, epochs=5, seed=42)
    print(f'Trained word2vec, vocab={len(w2v.wv)} ({time.time()-t0:.0f}s)')

    def embed(smi, dim=100):
        toks = morgan_sentence(smi) or []
        vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

    emb = np.vstack([embed(s) for s in all_smiles])
    w2v_feats = pd.DataFrame(emb, index=all_smiles, columns=[f'w2v_{i}' for i in range(emb.shape[1])])
    w2v_feats.to_pickle(w2v_cache)

mol_feats_v2 = mol_feats.join(w2v_feats)

# ---------------------------------------------------------------
# 3. Cross-property features: known OTHER target values for same molecule
# ---------------------------------------------------------------
wide = train.pivot_table(index='smiles', columns='target_type', values='target', aggfunc='mean')
wide = wide.reindex(columns=TARGETS)
wide.columns = [f'xf_{c}' for c in wide.columns]

def make_X(df, target_type, feats):
    X = feats.reindex(df['smiles'].values).reset_index(drop=True)
    xf = wide.reindex(df['smiles'].values).reset_index(drop=True).copy()
    xf[f'xf_{target_type}'] = np.nan  # never see own target as a feature
    return pd.concat([X, xf], axis=1)

LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15,
    min_child_samples=10, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=0.5,
    random_state=42, verbosity=-1
)

def cv_score(target_type, feats):
    sub = train[train['target_type'] == target_type].reset_index(drop=True)
    y = sub['target'].values
    groups = sub['smiles'].values
    X = make_X(sub, target_type, feats)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(sub))
    for tr_idx, va_idx in gkf.split(X, y, groups):
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X.iloc[va_idx])
    return r2_score(y, oof)

# ---------------------------------------------------------------
# 4. Per-target feature-set selection (does PI1M embedding help THIS target?)
# ---------------------------------------------------------------
print(f'\n{"target":6s} {"n":>5s} {"no w2v":>8s} {"+w2v":>8s} {"chosen":>8s}')
choice, best_r2 = {}, {}
for tt in TARGETS:
    n = int((train['target_type'] == tt).sum())
    r_prev = cv_score(tt, mol_feats)
    r_new = cv_score(tt, mol_feats_v2)
    choice[tt] = 'w2v' if r_new > r_prev else 'prev'
    best_r2[tt] = max(r_prev, r_new)
    print(f'{tt:6s} {n:5d} {r_prev:8.4f} {r_new:8.4f} {choice[tt]:>8s}')
print(f'{"MEAN":6s} {"":5s} {"":8s} {"":8s} {np.mean(list(best_r2.values())):8.4f}')

# ---------------------------------------------------------------
# 5. Final fit on full train (per-target best feature set), predict test
# ---------------------------------------------------------------
preds = np.zeros(len(test))
for tt in TARGETS:
    feats = mol_feats_v2 if choice[tt] == 'w2v' else mol_feats
    mask_tr = (train['target_type'] == tt).values
    Xtr = make_X(train[mask_tr].reset_index(drop=True), tt, feats)
    ytr = train.loc[mask_tr, 'target'].values
    mask_te = (test['target_type'] == tt).values
    Xte = make_X(test[mask_te].reset_index(drop=True), tt, feats)
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(Xtr, ytr)
    preds[mask_te] = model.predict(Xte)

submission = pd.DataFrame({'id': test['id'], 'target': preds})
submission.to_csv('/mnt/user-data/outputs/submission.csv', index=False)
print('\nSaved submission:', submission.shape)
