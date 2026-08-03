"""
AISEHack 2.0 Round 2 - Polymer Property Prediction
LightGBM + multi-task MLP, blended with the NB03 sibling-stacking model.

WHAT THIS SCRIPT DOES
----------------------
1. Featurizes every unique polymer SMILES (RDKit descriptors + 256-bit Morgan
   fingerprint), plus a word2vec embedding trained from scratch on PI1M.csv's
   ~1M unlabeled SMILES (no pretrained weights - satisfies the "no external
   pretrained models" rule; it's auxiliary data the competition itself ships).
2. Adds cross-property ("sibling") features: for a molecule with other known
   properties in train.csv, those values are used as extra input features
   when predicting a DIFFERENT property of the same molecule. This works
   because Round 2's test split is row-level random, not molecule-level, so
   a test row asking for e.g. `eps` very often has `nc`/`ei`/`eea`/`egb` for
   that same molecule already sitting in train.
3. Trains a tuned LightGBM (per-target Optuna-searched hyperparameters,
   included below and reusable via RETUNE=True) and a from-scratch
   multi-task MLP (pure numpy - no torch dependency) per target.
4. Blends those two with NB03's sibling-stacking model (a separate, more
   powerful implementation of the same cross-property idea, using explicit
   physics terms - see that notebook for its own methodology) via
   non-negative least squares on out-of-fold predictions.
5. Assembles the final submission: the blend applies ONLY to rows NB03's
   pipeline had to predict. Rows it answered via the Round-1 archive exact
   lookup (identified by diffing NB03's submission.csv against its raw
   pred_sibling.csv - wherever they differ, that row was overridden with a
   real measured value) are left completely untouched, since blending a
   perfect answer with an imperfect one only hurts.

REQUIRES, from NB03 (anrf-app5-v12.ipynb or your latest version), placed
alongside this script or in a Kaggle input dataset:
    oof_sibling.csv, pred_sibling.csv, submission.csv
Run NB03 first if you don't have these yet - this script blends WITH it,
it does not reproduce its Stage A / Stage B / archive-override logic.

Validated: mean OOF 0.896 (up from NB03's 0.893 alone), LB 0.902 (up from
NB03's 0.899 alone) on the run this was built from.
"""
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from gensim.models import Word2Vec
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.optimize as sopt
import lightgbm as lgb

RDLogger.DisableLog('rdApp.*')
TARGETS = ['egc', 'egb', 'ei', 'eea', 'eps', 'nc', 'tg']
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}
SEED = 42
RETUNE = False   # True re-runs Optuna from scratch per target (slow - ~15-40 min); False reuses BEST_PARAMS below

# =====================================================================
# 1. Data loading - works on Kaggle (/kaggle/input) or a local folder
# =====================================================================
def find_file(*name_patterns, roots=None):
    roots = roots or ['/kaggle/input', '.', './data', './artifacts', '..']
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat in name_patterns:
            hits = glob.glob(os.path.join(root, '**', pat), recursive=True)
            if hits:
                return hits[0]
    return None

DATA_DIR = os.environ.get('DATA_DIR', '.')  # override with your own path if needed

def load_required(name_patterns, what):
    p = find_file(*name_patterns)
    if p is None:
        raise FileNotFoundError(
            f"Couldn't find {what} (looked for {name_patterns} under /kaggle/input, ., ./data, "
            f"./artifacts, ..). Set DATA_DIR or place the file in one of those locations."
        )
    return p

train = pd.read_csv(load_required(['train.csv'], 'train.csv'))
test = pd.read_csv(load_required(['test.csv'], 'test.csv'))
pi1m_path = find_file('PI1M.csv')
sib_oof_path = find_file('oof_sibling.csv')
sib_pred_path = find_file('pred_sibling.csv')
sib_sub_path = find_file('submission.csv', roots=['/kaggle/input', '.', './nb03_output', './artifacts'])
for name, p in [('PI1M.csv', pi1m_path), ('oof_sibling.csv', sib_oof_path),
                ('pred_sibling.csv', sib_pred_path), ('NB03 submission.csv', sib_sub_path)]:
    if p is None:
        raise FileNotFoundError(f"Couldn't find {name}. Run NB03 first and place its outputs "
                                 f"alongside this script (or PI1M.csv from the competition data).")
print(f'train {train.shape}  test {test.shape}')
print(f'PI1M {pi1m_path}\nsibling OOF {sib_oof_path}\nsibling pred {sib_pred_path}\nsibling submission {sib_sub_path}')

CACHE = Path('./cache')
CACHE.mkdir(exist_ok=True)

# =====================================================================
# 2. Molecular features: RDKit descriptors + Morgan fingerprint (cached)
# =====================================================================
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

all_smiles = pd.unique(pd.concat([train['smiles'], test['smiles']], ignore_index=True))
desc_cache = CACHE / 'mol_feats.pkl'
if desc_cache.exists():
    mol_feats = pd.read_pickle(desc_cache)
else:
    t0 = time.time()
    mol_feats = compute_mol_features(all_smiles)
    mol_feats.to_pickle(desc_cache)
    print(f'computed descriptors+FP for {len(all_smiles)} molecules ({time.time()-t0:.0f}s)')

# =====================================================================
# 3. PI1M word2vec embeddings, trained from scratch (cached)
# =====================================================================
def morgan_sentence(smi, radius=1):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprint(mol, radius)
    return [str(k) for k in fp.GetNonzeroElements().keys()]

w2v_cache = CACHE / 'w2v_embeddings.pkl'
if w2v_cache.exists():
    w2v_feats = pd.read_pickle(w2v_cache)
else:
    t0 = time.time()
    pi1m = pd.read_csv(pi1m_path)
    sentences = [s for s in (morgan_sentence(x) for x in pi1m['SMILES']) if s]
    print(f'PI1M corpus: {len(sentences)} sentences ({time.time()-t0:.0f}s)')
    t0 = time.time()
    w2v = Word2Vec(sentences=sentences, vector_size=100, window=10, min_count=3,
                    sg=0, workers=1, epochs=5, seed=SEED)
    print(f'word2vec trained, vocab={len(w2v.wv)} ({time.time()-t0:.0f}s)')

    def embed(smi, dim=100):
        toks = morgan_sentence(smi) or []
        vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

    emb = np.vstack([embed(s) for s in all_smiles])
    w2v_feats = pd.DataFrame(emb, index=all_smiles, columns=[f'w2v_{i}' for i in range(emb.shape[1])])
    w2v_feats.to_pickle(w2v_cache)

# Empirically, PI1M embeddings help egc/egb/tg but hurt the smallest targets
# (adds noise faster than signal below ~300 rows) - chosen via CV.
W2V_HELPS = {'egc': True, 'egb': True, 'ei': False, 'eea': False, 'eps': False, 'nc': False, 'tg': True}
mol_feats_v2 = mol_feats.join(w2v_feats)

# =====================================================================
# 4. Cross-property ("sibling") features
# =====================================================================
wide = train.pivot_table(index='smiles', columns='target_type', values='target', aggfunc='mean')
wide = wide.reindex(columns=TARGETS)
wide.columns = [f'xf_{c}' for c in wide.columns]

def make_X_lgb(df, target_type, feats):
    X = feats.reindex(df['smiles'].values).reset_index(drop=True)
    xf = wide.reindex(df['smiles'].values).reset_index(drop=True).copy()
    xf[f'xf_{target_type}'] = np.nan  # never see own target
    return pd.concat([X, xf], axis=1)

# =====================================================================
# 5. LightGBM - tuned hyperparameters (from a 20-35 trial Optuna search
#    per target; set RETUNE=True at the top to redo it)
# =====================================================================
BEST_PARAMS = {
    "ei":  {"n_estimators": 434, "learning_rate": 0.023430670206249088, "num_leaves": 11, "min_child_samples": 8,  "subsample": 0.8499239395288354, "colsample_bytree": 0.7575967310909005, "reg_alpha": 0.0013970734935405218, "reg_lambda": 0.3550450967004018},
    "eea": {"n_estimators": 383, "learning_rate": 0.05323362454332515,  "num_leaves": 10, "min_child_samples": 15, "subsample": 0.8321872195886282, "colsample_bytree": 0.3012679800368632, "reg_alpha": 0.2478550395172973,   "reg_lambda": 0.07156223871702778},
    "eps": {"n_estimators": 436, "learning_rate": 0.022553010526064232, "num_leaves": 24, "min_child_samples": 8,  "subsample": 0.7252641487033147, "colsample_bytree": 0.8516225490967527, "reg_alpha": 0.06317556406207281,  "reg_lambda": 0.977024079077265},
    "nc":  {"n_estimators": 325, "learning_rate": 0.027720204535559637, "num_leaves": 32, "min_child_samples": 9,  "subsample": 0.8354881618141738, "colsample_bytree": 0.4625351124882012, "reg_alpha": 0.14780056216288004,  "reg_lambda": 0.010773780497341628},
    "egb": {"n_estimators": 222, "learning_rate": 0.03347776308515933,  "num_leaves": 19, "min_child_samples": 11, "subsample": 0.8059264473611898, "colsample_bytree": 0.3976457024564293, "reg_alpha": 0.01474275315991467,  "reg_lambda": 0.029204338471814112},
    "egc": {"n_estimators": 404, "learning_rate": 0.030737441977957157, "num_leaves": 13, "min_child_samples": 9,  "subsample": 0.9023871185418043, "colsample_bytree": 0.916286109921024,  "reg_alpha": 0.42588427892532865,  "reg_lambda": 0.015413983953545984},
    "tg":  {"n_estimators": 212, "learning_rate": 0.08927180304353628,  "num_leaves": 36, "min_child_samples": 31, "subsample": 0.5780093202212182, "colsample_bytree": 0.40919616423534183,"reg_alpha": 0.0017073967431528124,"reg_lambda": 2.9154431891537547},
}

def tune_one_target(tt, n_trials=35):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feats = mol_feats_v2 if W2V_HELPS[tt] else mol_feats

    def cv_r2(params):
        sub = train[train['target_type'] == tt].reset_index(drop=True)
        y, groups = sub['target'].values, sub['smiles'].values
        X = make_X_lgb(sub, tt, feats)
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(X, y, groups):
            m = lgb.LGBMRegressor(**params, random_state=SEED, verbosity=-1)
            m.fit(X.iloc[tr], y[tr])
            oof[va] = m.predict(X.iloc[va])
        return r2_score(y, oof)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 800),
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int('num_leaves', 4, 63),
            min_child_samples=trial.suggest_int('min_child_samples', 3, 50),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        )
        return cv_r2(params)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

if RETUNE:
    print('RETUNE=True: re-running Optuna per target (this takes a while)...')
    BEST_PARAMS = {tt: tune_one_target(tt) for tt in TARGETS}
    json.dump(BEST_PARAMS, open(CACHE / 'retuned_params.json', 'w'), indent=2)

LGB_KW = {tt: dict(**BEST_PARAMS[tt], subsample_freq=1, random_state=SEED, verbosity=-1) for tt in TARGETS}

# =====================================================================
# 6. Multi-task MLP - pure numpy, no torch dependency. One shared body,
#    7 output heads, masked loss (each row only trains its own target's
#    head). Input is PCA-compressed descriptors/FP/w2v + cross features.
# =====================================================================
N_PCA = 64
imp_pca = SimpleImputer(strategy='median')
scaler_pca = StandardScaler()
pca = PCA(n_components=N_PCA, random_state=SEED)
Xp = pca.fit_transform(scaler_pca.fit_transform(imp_pca.fit_transform(mol_feats_v2.values)))
pca_feats = pd.DataFrame(Xp, index=mol_feats_v2.index, columns=[f'pca_{i}' for i in range(N_PCA)])
print(f'MLP input PCA: {N_PCA} components explain {pca.explained_variance_ratio_.sum():.3f} of variance')

def make_X_mlp(df):
    Xp_ = pca_feats.reindex(df['smiles'].values).values
    xf_ = wide.reindex(df['smiles'].values).values
    return np.hstack([Xp_, xf_])

def mask_self_target(X, task_idx, n_pca=N_PCA):
    X = X.copy()
    X[np.arange(len(X)), n_pca + task_idx] = np.nan
    return X

class MultiTaskMLP:
    """Shared 2-hidden-layer body, one linear output per target. Hand-rolled
    (forward/backward/Adam) so this has zero deep-learning-framework
    dependency - only numpy."""
    def __init__(self, in_dim, hidden1=64, hidden2=32, out_dim=7, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, hidden1) * np.sqrt(2.0 / in_dim); self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1); self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, out_dim) * np.sqrt(2.0 / hidden2); self.b3 = np.zeros(out_dim)
        self.names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        self.m = {n: np.zeros_like(getattr(self, n)) for n in self.names}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in self.names}
        self.t = 0

    def forward(self, X, task_idx, training=False, dropout=0.3, rng=None):
        z1 = X @ self.W1 + self.b1; a1 = np.maximum(0, z1); mask1 = None
        if training:
            mask1 = (rng.rand(*a1.shape) > dropout) / (1 - dropout); a1 = a1 * mask1
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(0, z2); mask2 = None
        if training:
            mask2 = (rng.rand(*a2.shape) > dropout) / (1 - dropout); a2 = a2 * mask2
        out = a2 @ self.W3 + self.b3
        pred = out[np.arange(len(out)), task_idx]
        return pred, (X, z1, a1, mask1, z2, a2, mask2, out)

    def backward(self, cache, task_idx, y, weight_decay=1e-4):
        X, z1, a1, mask1, z2, a2, mask2, out = cache
        N = X.shape[0]
        pred = out[np.arange(N), task_idx]
        dout = np.zeros_like(out); dout[np.arange(N), task_idx] = 2 * (pred - y) / N
        dW3 = a2.T @ dout + weight_decay * self.W3; db3 = dout.sum(axis=0)
        da2 = dout @ self.W3.T
        if mask2 is not None: da2 = da2 * mask2
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2 + weight_decay * self.W2; db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        if mask1 is not None: da1 = da1 * mask1
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1 + weight_decay * self.W1; db1 = dz1.sum(axis=0)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for n, g in grads.items():
            self.m[n] = beta1 * self.m[n] + (1 - beta1) * g
            self.v[n] = beta2 * self.v[n] + (1 - beta2) * (g ** 2)
            mhat = self.m[n] / (1 - beta1 ** self.t); vhat = self.v[n] / (1 - beta2 ** self.t)
            setattr(self, n, getattr(self, n) - lr * mhat / (np.sqrt(vhat) + eps))

    def predict(self, X, task_idx):
        pred, _ = self.forward(X, task_idx, training=False)
        return pred

def train_mlp(Xtr, task_tr, ytr, Xva, task_va, yva, epochs=300, patience=30, batch_size=128, seed=0):
    model = MultiTaskMLP(in_dim=Xtr.shape[1], hidden1=64, hidden2=32, out_dim=len(TARGETS), seed=seed)
    rng = np.random.RandomState(seed)
    n = len(Xtr)
    best_val, best_snap, bad = np.inf, None, 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            pred, cache = model.forward(Xtr[idx], task_tr[idx], training=True, dropout=0.3, rng=rng)
            grads = model.backward(cache, task_tr[idx], ytr[idx], weight_decay=1e-4)
            model.adam_step(grads, lr=1e-3)
        val_loss = np.mean((model.predict(Xva, task_va) - yva) ** 2)
        if val_loss < best_val - 1e-6:
            best_val, best_snap, bad = val_loss, {n_: getattr(model, n_).copy() for n_ in model.names}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    for n_, v in best_snap.items():
        setattr(model, n_, v)
    return model

# =====================================================================
# 7. Out-of-fold predictions for LightGBM and the MLP (GroupKFold by
#    SMILES, so no molecule leaks across a fold's train/val split)
# =====================================================================
def lgb_oof_for_target(tt):
    feats = mol_feats_v2 if W2V_HELPS[tt] else mol_feats
    sub = train[train['target_type'] == tt].reset_index(drop=True)
    y, groups = sub['target'].values, sub['smiles'].values
    X = make_X_lgb(sub, tt, feats)
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = lgb.LGBMRegressor(**LGB_KW[tt])
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
    return oof

print('\nGenerating LightGBM OOF predictions...')
lgb_oof = np.zeros(len(train))
for tt in TARGETS:
    mask = (train['target_type'] == tt).values
    oof = lgb_oof_for_target(tt)
    lgb_oof[mask] = oof
    print(f'  {tt:6s} R2={r2_score(train.loc[mask, "target"], oof):.4f}')

print('\nGenerating multi-task MLP OOF predictions (GroupKFold, joint across targets)...')
mlp_oof = np.zeros(len(train))
for fold, (tr_idx, va_idx) in enumerate(GroupKFold(5).split(train, groups=train['smiles'])):
    tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
    task_tr = tr_df['target_type'].map(TARGET_IDX).values
    task_va = va_df['target_type'].map(TARGET_IDX).values
    Xtr_raw = mask_self_target(make_X_mlp(tr_df), task_tr)
    Xva_raw = mask_self_target(make_X_mlp(va_df), task_va)
    t_mean = tr_df.groupby('target_type')['target'].mean().reindex(TARGETS).values
    t_std = tr_df.groupby('target_type')['target'].std().reindex(TARGETS).values
    ytr_norm = (tr_df['target'].values - t_mean[task_tr]) / t_std[task_tr]
    yva_norm = (va_df['target'].values - t_mean[task_va]) / t_std[task_va]
    imp2, sc2 = SimpleImputer(strategy='median'), StandardScaler()
    Xtr2 = sc2.fit_transform(imp2.fit_transform(Xtr_raw))
    Xva2 = sc2.transform(imp2.transform(Xva_raw))
    model = train_mlp(Xtr2, task_tr, ytr_norm, Xva2, task_va, yva_norm, seed=fold)
    pred_norm = model.predict(Xva2, task_va)
    mlp_oof[va_idx] = pred_norm * t_std[task_va] + t_mean[task_va]
    print(f'  fold {fold} done')
for tt in TARGETS:
    mask = (train['target_type'] == tt).values
    print(f'  {tt:6s} R2={r2_score(train.loc[mask, "target"], mlp_oof[mask]):.4f}')

# =====================================================================
# 8. Blend with NB03's sibling-stacking OOF via non-negative least
#    squares (per target - lets each target pick its own mix)
# =====================================================================
sib_oof = pd.read_csv(sib_oof_path)  # smiles, target_type, target, pred
m = pd.DataFrame({'smiles': train['smiles'], 'target_type': train['target_type'],
                   'target': train['target'], 'mine_lgb': lgb_oof, 'mine_mlp': mlp_oof})
m = m.merge(sib_oof[['smiles', 'target_type', 'pred']].rename(columns={'pred': 'sibling'}),
            on=['smiles', 'target_type'], how='inner')

print(f'\n{"target":6s} {"n":>5s} {"blend R2":>9s}  weights(lgb,mlp,sibling)')
blend_weights, blend_r2 = {}, {}
for tt in TARGETS:
    g = m[m['target_type'] == tt]
    y = g['target'].values
    P = g[['mine_lgb', 'mine_mlp', 'sibling']].values
    w, _ = sopt.nnls(P, y)
    r = r2_score(y, P @ w)
    blend_weights[tt] = w.tolist()
    blend_r2[tt] = r
    print(f'{tt:6s} {len(g):5d} {r:9.4f}  {np.round(w, 3)}')
print(f'MEAN blended OOF: {np.mean(list(blend_r2.values())):.4f}')

# =====================================================================
# 9. Final refit on full train, predict test (LightGBM + MLP)
# =====================================================================
print('\nFinal refit on full train...')
lgb_test_pred = {}
for tt in TARGETS:
    feats = mol_feats_v2 if W2V_HELPS[tt] else mol_feats
    mask_tr = (train['target_type'] == tt).values
    mask_te = (test['target_type'] == tt).values
    Xtr = make_X_lgb(train[mask_tr].reset_index(drop=True), tt, feats)
    Xte = make_X_lgb(test[mask_te].reset_index(drop=True), tt, feats)
    m_ = lgb.LGBMRegressor(**LGB_KW[tt])
    m_.fit(Xtr, train.loc[mask_tr, 'target'].values)
    lgb_test_pred[tt] = (mask_te, m_.predict(Xte))

t_mean_full = train.groupby('target_type')['target'].mean().reindex(TARGETS).values
t_std_full = train.groupby('target_type')['target'].std().reindex(TARGETS).values
task_train_full = train['target_type'].map(TARGET_IDX).values
Xtr_raw = mask_self_target(make_X_mlp(train), task_train_full)
imp3, sc3 = SimpleImputer(strategy='median'), StandardScaler()
Xtr3 = sc3.fit_transform(imp3.fit_transform(Xtr_raw))
ytr3 = (train['target'].values - t_mean_full[task_train_full]) / t_std_full[task_train_full]
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(Xtr3))
n_val = max(200, len(Xtr3) // 10)
va_i, tr_i = perm[:n_val], perm[n_val:]
final_mlp = train_mlp(Xtr3[tr_i], task_train_full[tr_i], ytr3[tr_i],
                       Xtr3[va_i], task_train_full[va_i], ytr3[va_i], seed=99)
task_test = test['target_type'].map(TARGET_IDX).values
Xte_raw = mask_self_target(make_X_mlp(test), task_test)
Xte3 = sc3.transform(imp3.transform(Xte_raw))
mlp_test_norm = final_mlp.predict(Xte3, task_test)
mlp_test_pred = mlp_test_norm * t_std_full[task_test] + t_mean_full[task_test]

# =====================================================================
# 10. Assemble the final submission
#     - blend LightGBM + MLP + NB03's sibling prediction on rows NB03
#       actually had to predict
#     - leave rows NB03 answered via the Round-1 archive exact-match
#       lookup COMPLETELY untouched (those are ground truth - identified
#       by diffing its submission.csv against its raw pred_sibling.csv)
# =====================================================================
pred_sib = pd.read_csv(sib_pred_path)          # id, smiles, target_type, pred
their_sub = pd.read_csv(sib_sub_path)          # id, target (post-override)
assert (pred_sib['id'].values == test['id'].values).all()
assert (their_sub['id'].values == test['id'].values).all()

blended = np.zeros(len(test))
for tt in TARGETS:
    mask, lgb_pred = lgb_test_pred[tt]
    mlp_pred = mlp_test_pred[mask]
    sib_pred = pred_sib.loc[mask, 'pred'].values
    w_lgb, w_mlp, w_sib = blend_weights[tt]
    blended[mask] = w_lgb * lgb_pred + w_mlp * mlp_pred + w_sib * sib_pred

overridden = ~np.isclose(their_sub['target'].values, pred_sib['pred'].values, atol=1e-6)
print(f'\narchive-overridden rows kept as ground truth: {overridden.sum()}/{len(test)}')

final = np.where(overridden, their_sub['target'].values, blended)
out = pd.DataFrame({'id': test['id'], 'target': final}).sort_values('id')
assert out['target'].notna().all()
out.to_csv('final_submission.csv', index=False)
print('Saved final_submission.csv:', out.shape)
print(out.head())
