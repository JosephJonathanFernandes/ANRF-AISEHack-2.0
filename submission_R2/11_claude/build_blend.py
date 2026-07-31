"""
AISEHack 2.0 Round 2 - LightGBM + multi-task MLP blend
This is a different direction from Nihaal's AttentiveFP GNN: instead of a
graph model, it adds a hand-rolled (pure numpy, no torch needed) multi-task
neural net as a second, structurally-different model, then blends it with
the LightGBM pipeline using per-target weights chosen by cross-validation.

Honest summary of what happened when building this:
- The MLP alone is WORSE than LightGBM on every target (mean CV 0.788 vs
  0.854) - unsurprising, gradient-boosted trees are consistently strong on
  tabular data at this dataset size, well-documented in ML literature.
- But blended with LightGBM (mostly LightGBM weight, ~10-25% MLP), EVERY
  target improves. Mean CV goes from 0.854 to ~0.858.
- The blend weights below are picked by grid search on out-of-fold
  predictions (never touches the test set), so they should generalize
  the same way the earlier 0.854 CV number did (it landed within 0.001-0.002
  of the real leaderboard score twice in a row).
- Cross-property features use ONE lookup table built from the full train
  set, not rebuilt per CV fold. This is deliberate, not an oversight: a
  molecule's other-property value is a legitimate, always-available input
  feature (exactly like any RDKit descriptor), not a peek at the row being
  predicted's own label - it's exactly as available for a real test
  molecule as for a CV validation one. This is the same design validated
  by the 0.854 CV / 0.852 then 0.855 LB match in the earlier pipeline.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb

TARGETS = ['egc', 'egb', 'ei', 'eea', 'eps', 'nc', 'tg']
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}
N_PCA = 64

train = pd.read_csv('/mnt/user-data/uploads/train.csv')
test = pd.read_csv('/mnt/user-data/uploads/test.csv')
mol_feats = pd.read_pickle('/home/claude/mol_feats_cache.pkl')      # descriptors + Morgan FP, see build_model.py
w2v_feats = pd.read_pickle('/home/claude/w2v_embeddings.pkl')       # PI1M word2vec embeddings, see build_model.py
mol_feats_v2 = mol_feats.join(w2v_feats)

# =================================================================
# Part 1: LightGBM (same recipe as build_model.py)
# =================================================================
def build_wide(df):
    w = df.pivot_table(index='smiles', columns='target_type', values='target', aggfunc='mean')
    w = w.reindex(columns=TARGETS)
    w.columns = [f'xf_{c}' for c in w.columns]
    return w

def make_X_lgb(df, target_type, feats, wide_table):
    X = feats.reindex(df['smiles'].values).reset_index(drop=True)
    xf = wide_table.reindex(df['smiles'].values).reset_index(drop=True).copy()
    xf[f'xf_{target_type}'] = np.nan
    return pd.concat([X, xf], axis=1)

LGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_alpha=0.5,
                   reg_lambda=0.5, random_state=42, verbosity=-1)

def lgb_cv_oof(target_type, feats, wide_table):
    sub = train[train['target_type'] == target_type].reset_index(drop=True)
    idx_global = train.index[train['target_type'] == target_type].values
    y, groups = sub['target'].values, sub['smiles'].values
    gkf = GroupKFold(5)
    oof = np.zeros(len(sub))
    for tr, va in gkf.split(sub, y, groups):
        Xtr = make_X_lgb(sub.iloc[tr], target_type, feats, wide_table)
        Xva = make_X_lgb(sub.iloc[va], target_type, feats, wide_table)
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(Xtr, y[tr])
        oof[va] = m.predict(Xva)
    full_oof = np.zeros(len(train))
    full_oof[idx_global] = oof
    return full_oof, idx_global

# One global cross-feature table built from all of train (not leakage - see
# docstring note above; a molecule's other-property value is a legitimate,
# always-available feature, not a peek at the row's own label).
GLOBAL_WIDE = build_wide(train)

print('Selecting per-target feature set (descriptors+FP vs +PI1M embeddings)...')
lgb_choice, lgb_oof = {}, np.zeros(len(train))
for tt in TARGETS:
    oof_prev, idx = lgb_cv_oof(tt, mol_feats, GLOBAL_WIDE)
    oof_w2v, _ = lgb_cv_oof(tt, mol_feats_v2, GLOBAL_WIDE)
    mask = train['target_type'].values == tt
    r_prev = r2_score(train['target'].values[mask], oof_prev[mask])
    r_w2v = r2_score(train['target'].values[mask], oof_w2v[mask])
    lgb_choice[tt] = 'w2v' if r_w2v > r_prev else 'prev'
    lgb_oof[mask] = oof_w2v[mask] if r_w2v > r_prev else oof_prev[mask]
    print(f'  {tt:6s} prev={r_prev:.4f} w2v={r_w2v:.4f} -> {lgb_choice[tt]}')

# =================================================================
# Part 2: multi-task MLP (pure numpy, PCA-compressed features)
# =================================================================
class MultiTaskMLP:
    def __init__(self, in_dim, hidden1=64, hidden2=32, out_dim=7, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, hidden1) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, out_dim) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(out_dim)
        self.names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        self.m = {n: np.zeros_like(getattr(self, n)) for n in self.names}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in self.names}
        self.t = 0

    def forward(self, X, task_idx, training=False, dropout=0.3, rng=None):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        mask1 = None
        if training:
            mask1 = (rng.rand(*a1.shape) > dropout) / (1 - dropout)
            a1 = a1 * mask1
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        mask2 = None
        if training:
            mask2 = (rng.rand(*a2.shape) > dropout) / (1 - dropout)
            a2 = a2 * mask2
        out = a2 @ self.W3 + self.b3
        pred = out[np.arange(len(out)), task_idx]
        return pred, (X, z1, a1, mask1, z2, a2, mask2, out)

    def backward(self, cache, task_idx, y, weight_decay=1e-4):
        X, z1, a1, mask1, z2, a2, mask2, out = cache
        N = X.shape[0]
        pred = out[np.arange(N), task_idx]
        dout = np.zeros_like(out)
        dout[np.arange(N), task_idx] = 2 * (pred - y) / N
        dW3 = a2.T @ dout + weight_decay * self.W3
        db3 = dout.sum(axis=0)
        da2 = dout @ self.W3.T
        if mask2 is not None:
            da2 = da2 * mask2
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2 + weight_decay * self.W2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        if mask1 is not None:
            da1 = da1 * mask1
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1 + weight_decay * self.W1
        db1 = dz1.sum(axis=0)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for n, g in grads.items():
            self.m[n] = beta1 * self.m[n] + (1 - beta1) * g
            self.v[n] = beta2 * self.v[n] + (1 - beta2) * (g ** 2)
            mhat = self.m[n] / (1 - beta1 ** self.t)
            vhat = self.v[n] / (1 - beta2 ** self.t)
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

print('\nPCA-compressing descriptor+FP+PI1M features for the MLP...')
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
pca = PCA(n_components=N_PCA, random_state=42)
Xp = pca.fit_transform(scaler.fit_transform(imp.fit_transform(mol_feats_v2.values)))
print(f'{N_PCA} components explain {pca.explained_variance_ratio_.sum():.3f} of variance')
pca_feats = pd.DataFrame(Xp, index=mol_feats_v2.index, columns=[f'pca_{i}' for i in range(N_PCA)])

def make_X_mlp(df, wide_table):
    Xp_ = pca_feats.reindex(df['smiles'].values).values
    xf_ = wide_table.reindex(df['smiles'].values).values
    return np.hstack([Xp_, xf_])

def mask_self_target(X, task_idx, n_pca=N_PCA):
    X = X.copy()
    X[np.arange(len(X)), n_pca + task_idx] = np.nan
    return X

print('Running MLP GroupKFold CV...')
gkf = GroupKFold(n_splits=5)
mlp_oof = np.zeros(len(train))
for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train['smiles'])):
    tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
    task_tr = tr_df['target_type'].map(TARGET_IDX).values
    task_va = va_df['target_type'].map(TARGET_IDX).values
    Xtr_raw = mask_self_target(make_X_mlp(tr_df, GLOBAL_WIDE), task_tr)
    Xva_raw = mask_self_target(make_X_mlp(va_df, GLOBAL_WIDE), task_va)

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

# =================================================================
# Part 3: per-target blend weight, chosen by CV (grid search on OOF)
# =================================================================
print(f'\n{"target":6s} {"n":>5s} {"LGBM":>8s} {"MLP":>8s} {"blend":>8s} {"w_lgbm":>7s}')
blend_w, blend_r2 = {}, {}
for tt in TARGETS:
    mask = (train['target_type'] == tt).values
    y = train['target'].values[mask]
    r_lgbm = r2_score(y, lgb_oof[mask])
    r_mlp = r2_score(y, mlp_oof[mask])
    best_w, best_r = 1.0, r_lgbm
    for w in np.linspace(0, 1, 41):
        r = r2_score(y, w * lgb_oof[mask] + (1 - w) * mlp_oof[mask])
        if r > best_r:
            best_r, best_w = r, w
    blend_w[tt], blend_r2[tt] = best_w, best_r
    print(f'{tt:6s} {mask.sum():5d} {r_lgbm:8.4f} {r_mlp:8.4f} {best_r:8.4f} {best_w:7.2f}')
print(f'{"MEAN":6s} {"":5s} {"":8s} {"":8s} {np.mean(list(blend_r2.values())):8.4f}')

# =================================================================
# Part 4: final refit on full train, predict test, blend, submit
# =================================================================
print('\nFinal refit...')
final_wide = GLOBAL_WIDE
lgb_test_pred = np.zeros(len(test))
for tt in TARGETS:
    feats = mol_feats_v2 if lgb_choice[tt] == 'w2v' else mol_feats
    mask_tr = (train['target_type'] == tt).values
    Xtr = make_X_lgb(train[mask_tr].reset_index(drop=True), tt, feats, final_wide)
    Xte = make_X_lgb(test[test['target_type'] == tt].reset_index(drop=True), tt, feats, final_wide)
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(Xtr, train.loc[mask_tr, 'target'].values)
    lgb_test_pred[(test['target_type'] == tt).values] = m.predict(Xte)

t_mean_full = train.groupby('target_type')['target'].mean().reindex(TARGETS).values
t_std_full = train.groupby('target_type')['target'].std().reindex(TARGETS).values
task_train_full = train['target_type'].map(TARGET_IDX).values
Xtr_raw = mask_self_target(make_X_mlp(train, final_wide), task_train_full)
imp3, sc3 = SimpleImputer(strategy='median'), StandardScaler()
Xtr3 = sc3.fit_transform(imp3.fit_transform(Xtr_raw))
ytr3 = (train['target'].values - t_mean_full[task_train_full]) / t_std_full[task_train_full]

rng = np.random.RandomState(42)
perm = rng.permutation(len(Xtr3))
n_val = max(200, len(Xtr3) // 10)
va_i, tr_i = perm[:n_val], perm[n_val:]
final_mlp = train_mlp(Xtr3[tr_i], task_train_full[tr_i], ytr3[tr_i],
                       Xtr3[va_i], task_train_full[va_i], ytr3[va_i], seed=99)

task_test = test['target_type'].map(TARGET_IDX).values
Xte_raw = mask_self_target(make_X_mlp(test, final_wide), task_test)
Xte3 = sc3.transform(imp3.transform(Xte_raw))
mlp_test_norm = final_mlp.predict(Xte3, task_test)
mlp_test_pred = mlp_test_norm * t_std_full[task_test] + t_mean_full[task_test]

w_per_row = np.array([blend_w[tt] for tt in test['target_type']])
final_pred = w_per_row * lgb_test_pred + (1 - w_per_row) * mlp_test_pred

submission = pd.DataFrame({'id': test['id'], 'target': final_pred})
submission.to_csv('/mnt/user-data/outputs/submission.csv', index=False)
print('\nSaved submission:', submission.shape)
