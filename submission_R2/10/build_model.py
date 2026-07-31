"""
AISEHack 2.0 Round 2 - Polymer Property Prediction  (submission_R2 / 10)
=========================================================================
v10: "Best of All Worlds" - Combining Joseph + Nihaal + Shivesh approaches

KEY INNOVATIONS (in order of expected impact):

1. [SHIVESH] SIBLING MASKING (mask_p=0.3):
   During training, randomly zero out cross-property features with prob=0.3.
   This forces the model to be robust when siblings are missing at test time,
   DIRECTLY fixing the CV/LB distribution shift that caused v8 to drop.

2. [SHIVESH] POLYMER-SPECIFIC HAND-CRAFTED FEATURES:
   16 physical descriptors encoding:
   - Backbone flexibility (rotatable bond density)
   - Aromatic stiffness (aromatic fraction)
   - H-bonding capacity (N+O fraction)
   - Polarizability per volume (Lorentz-Lorenz specific refraction)
   - Silicon fraction (low Tg driver)
   These are the PHYSICAL DRIVERS of the target properties, impossible to
   get from standard fingerprints alone.

3. [SHIVESH] ROUND 1 ARCHIVE DATA:
   The Round 1 archive contains 2,446 extra rows NOT in Round 2 train:
   - tg: +1,644 extra samples (massive, goes from 4143 -> 5787!)
   - egc: +804 extra samples (from 2028 -> 2832)
   This directly addresses the data scarcity problem for our best targets.

4. [SHIVESH] APPLY OVERRIDE:
   For test molecules that appear in training data for the SAME target_type,
   substitute the known ground-truth label directly. Free points!

5. [NIHAAL] MULTI-TASK NN BLEND:
   A feedforward neural network with a shared trunk + 7 per-target heads,
   trained with masked MSE loss (each row only sees its own target's gradient).
   This lets data-rich targets (tg, egc) help data-poor ones (eps, ei, eea, nc).
   We blend this with our best LightGBM at final prediction time.

6. [JOSEPH] PER-TARGET OPTIMAL FEATURE SETS (v6 best practice):
   Keep using individually tuned feature subsets for each target.
"""

import warnings
warnings.filterwarnings("ignore")

import os, gc, time, sys
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, rdFingerprintGenerator

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "Dataset")
ARCH_PATH   = os.path.join(BASE_DIR, "..", "..", "Round1_archive", "dataset", "train.csv")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_PATH  = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH   = os.path.join(DATASET_DIR, "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "submission.csv")

TARGETS = ["egc", "egb", "ei", "eea", "eps", "nc", "tg"]
SEED    = 42
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  |  PyTorch: {torch.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING + ROUND 1 ARCHIVE AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
train_r2 = pd.read_csv(TRAIN_PATH)
test      = pd.read_csv(TEST_PATH)

# Add Round 1 archive rows that don't overlap with Round 2 train
arch = pd.read_csv(ARCH_PATH)
r2_keys = set(zip(train_r2["smiles"], train_r2["target_type"]))
arch_extra = arch[~arch.apply(lambda r: (r["smiles"], r["target_type"]) in r2_keys, axis=1)].copy()
print(f"Round1 archive extra rows: {len(arch_extra)}  (by target: {dict(arch_extra['target_type'].value_counts())})")

train = pd.concat([train_r2, arch_extra], ignore_index=True).reset_index(drop=True)
print(f"Combined train: {len(train)} rows")

all_smiles = pd.unique(pd.concat([train["smiles"], test["smiles"]], ignore_index=True))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  POLYMER FEATURES (from Shivesh)
# ─────────────────────────────────────────────────────────────────────────────
def _star_ends(mol):
    ends = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            nb = a.GetNeighbors()
            if len(nb) != 1:
                return None
            b = mol.GetBondBetweenAtoms(a.GetIdx(), nb[0].GetIdx())
            ends.append((a.GetIdx(), nb[0].GetIdx(), b.GetBondType()))
    return ends if len(ends) == 2 else None


def build_cyclic(psmiles):
    """Build the cyclic (periodic) repeat unit: tail bonded back to head."""
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        return None
    ends = _star_ends(mol)
    if ends is None:
        return mol  # fallback to monomer
    rw = Chem.RWMol(mol)
    (s1, h1, _bt1), (s2, h2, bt2) = ends
    try:
        bt = bt2 if bt2 != Chem.BondType.AROMATIC else Chem.BondType.SINGLE
        rw.AddBond(h2, h1, bt)
        for idx in sorted([s1, s2], reverse=True):
            rw.RemoveAtom(idx)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return mol


POLY_NAMES = [
    "p_heavy", "p_nrot", "p_rot_dens", "p_arom_frac", "p_nring",
    "p_ring_dens", "p_sp3_frac", "p_mw", "p_mw_per_heavy", "p_tpsa",
    "p_tpsa_dens", "p_molmr", "p_spec_refr", "p_halo_frac", "p_no_frac", "p_si_frac",
]

def _polymer_features(mol):
    if mol is None:
        return np.full(16, np.nan, dtype=np.float32)
    try:
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        nrot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        narom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        nring = rdMolDescriptors.CalcNumRings(mol)
        nsp3  = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP3)
        mw    = Descriptors.MolWt(mol)
        tpsa  = rdMolDescriptors.CalcTPSA(mol)
        mr    = Descriptors.MolMR(mol)
        counts = {z: 0 for z in (9, 17, 35, 53, 14, 16, 7, 8)}
        for a in mol.GetAtoms():
            z = a.GetAtomicNum()
            if z in counts:
                counts[z] += 1
        halo = counts[9] + counts[17] + counts[35] + counts[53]
        return np.array([
            heavy, nrot, nrot / heavy, narom / heavy, nring, nring / heavy,
            nsp3 / heavy, mw, mw / heavy, tpsa, tpsa / heavy, mr,
            mr / max(mw, 1e-6), halo / heavy,
            (counts[7] + counts[8]) / heavy, counts[14] / heavy,
        ], dtype=np.float32)
    except Exception:
        return np.full(16, np.nan, dtype=np.float32)


print("Computing polymer features...")
POLY_CACHE_PATH = os.path.join(CACHE_DIR, "poly_feats_v10.pkl")
if os.path.exists(POLY_CACHE_PATH):
    poly_feats = pd.read_pickle(POLY_CACHE_PATH)
    print(f"  [cache] poly_feats {poly_feats.shape}")
else:
    rows = []
    for smi in all_smiles:
        cyc_mol = build_cyclic(smi)
        rows.append(_polymer_features(cyc_mol))
    poly_feats = pd.DataFrame(rows, index=all_smiles, columns=POLY_NAMES)
    poly_feats.to_pickle(POLY_CACHE_PATH)
    print(f"  Computed poly_feats {poly_feats.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MOLECULAR FINGERPRINTS & DESCRIPTORS (from v6)
# ─────────────────────────────────────────────────────────────────────────────
DESC_COLS = [name for name, _ in Descriptors.descList]

def _sanitize_df(df):
    for col in df.select_dtypes(include=["float32", "float64"]).columns:
        s = df[col]
        bad = ~(np.isfinite(s) & (s.abs() < 1e15))
        if bad.any():
            df[col] = s.where(~bad, other=np.nan)
    return df

mol_v3 = pd.read_pickle(os.path.join(BASE_DIR, "..", "6", "cache", "mol_feats_v3.pkl"))
maccs_cols   = [c for c in mol_v3.columns if c.startswith("maccs_")]
desc_feats   = _sanitize_df(mol_v3[DESC_COLS].copy())
maccs_feats  = mol_v3[maccs_cols].copy()
del mol_v3; gc.collect()

morgan256_feats = pd.read_pickle(os.path.join(BASE_DIR, "..", "6", "cache", "morgan256_v5.pkl"))

_svd_full = pd.read_pickle(os.path.join(BASE_DIR, "..", "6", "cache", "svd_embeddings_v3.pkl"))
pi1m64 = _svd_full.iloc[:, :64].copy()
pi1m64.columns = [f"svd_{i}" for i in range(64)]
del _svd_full; gc.collect()

# Build optimized per-target feature sets (from v6 analysis)
feats_A = desc_feats.join(morgan256_feats)                      # 473
feats_B = feats_A.join(pi1m64)                                   # 537
feats_C = feats_A.join(maccs_feats)                              # 640
feats_D = feats_C.join(pi1m64)                                   # 704

# Add polymer features to each
feats_Ap = feats_A.join(poly_feats)                              # 489
feats_Bp = feats_B.join(poly_feats)                              # 553
feats_Cp = feats_C.join(poly_feats)                              # 656
feats_Dp = feats_D.join(poly_feats)                              # 720

# v6 optimal per target (use poly-enhanced versions)
BEST_FEATS = {
    "eps": feats_Cp,   # C was best in v6 + polymer
    "ei":  feats_Bp,   # B was best in v6 + polymer
    "eea": feats_Cp,   # C was best in v6 + polymer
    "nc":  feats_Cp,   # C was best in v6 + polymer
    "egb": feats_Dp,   # D was best in v6 + polymer
    "egc": feats_Dp,   # D was best in v6 + polymer
    "tg":  feats_Cp,   # C was best in v6 + polymer
}
_HIGH = {"tg", "egc"}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CROSS-PROPERTY FEATURES WITH SIBLING MASKING (from Shivesh)
# ─────────────────────────────────────────────────────────────────────────────
wide = train.pivot_table(index="smiles", columns="target_type", values="target", aggfunc="mean")
wide = wide.reindex(columns=TARGETS)
wide.columns = [f"xf_{c}" for c in wide.columns]

# Also include wide from archive for test-time: use the FULL combined train
wide_full = pd.concat([train_r2, arch_extra]).pivot_table(
    index="smiles", columns="target_type", values="target", aggfunc="mean"
).reindex(columns=TARGETS)
wide_full.columns = [f"xf_{c}" for c in wide_full.columns]


def make_X(df, target_type, feats, mask_p=0.0, rng=None, xf_table=None):
    """Feature matrix with xf_ cross-property features.
    mask_p: probability of randomly masking each sibling feature (Shivesh's robustness trick).
    """
    if xf_table is None:
        xf_table = wide
    X  = feats.reindex(df["smiles"].values).reset_index(drop=True)
    xf = xf_table.reindex(df["smiles"].values).reset_index(drop=True).copy()
    xf[f"xf_{target_type}"] = np.nan  # never see own target

    # Sibling masking: randomly hide some cross-property values at training time
    if mask_p > 0.0 and rng is not None:
        for col in xf.columns:
            if col != f"xf_{target_type}":
                mask = rng.random(len(xf)) < mask_p
                xf.loc[mask, col] = np.nan

    return pd.concat([X, xf], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LIGHTGBM HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
def get_lgb_params(tt, n_estimators=None):
    base = dict(random_state=SEED, verbosity=-1, n_jobs=4,
                subsample=0.8, subsample_freq=1, reg_alpha=0.5)
    if tt in _HIGH:
        base.update(
            n_estimators=n_estimators or 3000,
            learning_rate=0.015,
            num_leaves=127,
            max_depth=8,
            min_child_samples=20,
            colsample_bytree=0.5,
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
# 6.  LIGHTGBM CROSS-VALIDATION WITH SIBLING MASKING
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== LightGBM CV with Sibling Masking (mask_p=0.30) ===")
MASK_P = 0.30

lgb_cv_scores = {}
lgb_oof_preds = {}
lgb_best_iters = {}
gkf = GroupKFold(n_splits=5)
rng = np.random.default_rng(SEED)

for tt in TARGETS:
    sub    = train[train["target_type"] == tt].reset_index(drop=True)
    y      = sub["target"].values
    groups = sub["smiles"].values
    feats  = BEST_FEATS[tt]
    params = get_lgb_params(tt)
    iters  = []
    oof    = np.zeros(len(sub))

    for tr_idx, va_idx in gkf.split(sub, y, groups):
        # Training: apply sibling masking
        Xtr = make_X(sub.iloc[tr_idx], tt, feats, mask_p=MASK_P, rng=rng)
        # Validation: NO masking (mimic test-time conditions)
        Xva = make_X(sub.iloc[va_idx], tt, feats, mask_p=0.0)
        ytr, yva = y[tr_idx], y[va_idx]

        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr,
              eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(Xva)
        iters.append(m.best_iteration_ if m.best_iteration_ > 0 else params["n_estimators"])

    r2 = r2_score(y, oof)
    lgb_cv_scores[tt] = r2
    lgb_oof_preds[tt] = (sub["smiles"].values, oof)
    lgb_best_iters[tt] = max(int(np.mean(iters) * 1.10), 50)
    print(f"  {tt:3s}: R2 = {r2:.4f}  |  avg_iter = {lgb_best_iters[tt]}")

lgb_mean_cv = np.mean(list(lgb_cv_scores.values()))
print(f"  MEAN LGB CV = {lgb_mean_cv:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MULTI-TASK NEURAL NETWORK (inspired by Nihaal + Shivesh)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Multi-Task Neural Network ===")

class MultiTaskNet(nn.Module):
    def __init__(self, d_in, n_out=7, trunk=(512, 256, 128), head_dim=64, p=0.25):
        super().__init__()
        layers, d = [], d_in
        for h in trunk:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, head_dim), nn.SiLU(), nn.Dropout(p / 2),
                nn.Linear(head_dim, 1)
            ) for _ in range(n_out)
        ])

    def forward(self, x):
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)  # (B, 7)


def build_nn_features():
    """Build a clean float feature matrix for the NN (desc + poly + morgan256)."""
    nn_feats_cache = os.path.join(CACHE_DIR, "nn_feats_v10.pkl")
    if os.path.exists(nn_feats_cache):
        return pd.read_pickle(nn_feats_cache)

    X = desc_feats.join(poly_feats).join(morgan256_feats)

    # Rank-normalize descriptors (Shivesh's trick for wildly different scales)
    desc_pp = desc_feats.copy()
    for col in desc_pp.columns:
        vals = desc_pp[col].dropna()
        if len(vals) > 1:
            order = vals.rank() / len(vals)
            desc_pp[col] = order.reindex(desc_pp.index)

    X = desc_pp.join(poly_feats).join(morgan256_feats)

    # Filter FP bits with near-zero or near-one variance (Shivesh's trick)
    fp_cols = [c for c in X.columns if c.startswith("mfp_") or c.startswith("Bit")]
    if fp_cols:
        on_rate = X[fp_cols].mean()
        keep_fp = on_rate[(on_rate > 0.005) & (on_rate < 0.995)].index.tolist()
        other_cols = [c for c in X.columns if c not in fp_cols]
        X = X[other_cols + keep_fp]

    X.to_pickle(nn_feats_cache)
    return X


nn_base = build_nn_features()

# Build molecule-indexed targets matrix
nn_uniq = sorted(set(train["smiles"]) | set(test["smiles"]))
nn_pos  = {s: i for i, s in enumerate(nn_uniq)}

Y_mat = np.full((len(nn_uniq), len(TARGETS)), np.nan, dtype=np.float32)
for _, row in train.iterrows():
    j = TARGETS.index(row["target_type"])
    Y_mat[nn_pos[row["smiles"]], j] = row["target"]

# Per-target z-score normalization
mu = np.nanmean(Y_mat, axis=0)
sd = np.nanstd(Y_mat, axis=0) + 1e-9
Yz = np.where(np.isfinite(Y_mat), (Y_mat - mu) / sd, np.nan)

# Train NN in a per-target OOF fashion using the same GroupKFold splits
print("Training Multi-Task NN...")
nn_dim = nn_base.shape[1]

def train_nn_fold(X_tr, Yz_tr, X_va, Yz_va, epochs=150, lr=1e-3, batch=256):
    """Train one fold's NN model."""
    model = MultiTaskNet(d_in=nn_dim).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Build tensor datasets
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    Y_tr_t = torch.tensor(Yz_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    Y_va_t = torch.tensor(Yz_va, dtype=torch.float32)

    best_val, best_state = float("inf"), None
    patience, patience_ct = 20, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(perm), batch):
            idx = perm[i:i+batch]
            xb, yb = X_tr_t[idx].to(DEVICE), Y_tr_t[idx].to(DEVICE)
            pred = model(xb)
            known = torch.isfinite(yb)
            loss = F.mse_loss(pred[known], yb[known]) if known.any() else pred.sum() * 0
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # Val loss
        model.eval()
        with torch.no_grad():
            pv = model(X_va_t.to(DEVICE)).cpu()
            known_v = torch.isfinite(Y_va_t)
            val_loss = F.mse_loss(pv[known_v], Y_va_t[known_v]).item() if known_v.any() else 0

        if val_loss < best_val - 1e-5:
            best_val, patience_ct = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= patience:
                break

    model.load_state_dict(best_state)
    return model


# Impute NaN in NN features with 0 (post normalization, 0 ≈ median)
nn_base_arr = nn_base.reindex(nn_uniq).fillna(0).values.astype(np.float32)

nn_oof_mat = np.full((len(nn_uniq), len(TARGETS)), np.nan, dtype=np.float32)

for tt_idx, tt in enumerate(TARGETS):
    mask   = np.isfinite(Y_mat[:, tt_idx])
    idxs   = np.where(mask)[0]
    smis   = [nn_uniq[i] for i in idxs]
    X_sub  = nn_base_arr[idxs]
    Yz_sub = Yz[idxs]  # all 7 targets for each mol in this target's training set
    groups_sub = smis

    gkf_nn = GroupKFold(n_splits=5)
    for tr_i, va_i in gkf_nn.split(X_sub, Yz_sub[:, tt_idx], groups_sub):
        X_tr, X_va = X_sub[tr_i], X_sub[va_i]
        Yz_tr, Yz_va = Yz_sub[tr_i], Yz_sub[va_i]
        model = train_nn_fold(X_tr, Yz_tr, X_va, Yz_va, epochs=120, lr=1e-3)
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(X_va, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        nn_oof_mat[idxs[va_i], tt_idx] = pv[:, tt_idx] * sd[tt_idx] + mu[tt_idx]

# Evaluate NN OOF per target
print("\nNN OOF R2 per target:")
nn_cv_scores = {}
for tt_idx, tt in enumerate(TARGETS):
    mask = np.isfinite(Y_mat[:, tt_idx]) & np.isfinite(nn_oof_mat[:, tt_idx])
    if mask.sum() > 10:
        r2 = r2_score(Y_mat[mask, tt_idx], nn_oof_mat[mask, tt_idx])
        nn_cv_scores[tt] = r2
        print(f"  {tt:3s}: R2 = {r2:.4f}")

nn_mean_cv = np.mean(list(nn_cv_scores.values()))
print(f"  MEAN NN CV = {nn_mean_cv:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FULL DATASET FINAL FIT (LGB + NN)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Final Fit: LGB + NN on full training data ===")

# Train final LGB models
lgb_models = {}
for tt in TARGETS:
    mask_tr = (train["target_type"] == tt).values
    Xtr = make_X(train[mask_tr].reset_index(drop=True), tt, BEST_FEATS[tt],
                 mask_p=0.0, xf_table=wide_full)  # use full table for final fit
    ytr = train.loc[mask_tr, "target"].values
    params = get_lgb_params(tt, n_estimators=lgb_best_iters[tt])
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr, ytr)
    lgb_models[tt] = m
    print(f"  LGB {tt}: trained on {Xtr.shape[0]} samples, {Xtr.shape[1]} features")

# Train final NN on full data
print("  NN: training final model on full data...")
X_all_nn  = nn_base_arr  # all molecules
Yz_all    = np.where(np.isfinite(Y_mat), (Y_mat - mu) / sd, np.nan)
n_all     = len(X_all_nn)
# Use 10% as held-out for early stopping
np.random.seed(SEED)
perm      = np.random.permutation(n_all)
n_val     = max(200, n_all // 10)
va_idx_f  = perm[:n_val]
tr_idx_f  = perm[n_val:]
final_nn  = train_nn_fold(X_all_nn[tr_idx_f], Yz_all[tr_idx_f],
                           X_all_nn[va_idx_f], Yz_all[va_idx_f], epochs=150)
print("  NN: final model trained.")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PREDICT ON TEST + APPLY OVERRIDE (Shivesh)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Predicting on test ===")

# Blend weight: LGB has been consistently stronger, but NN adds diversity
LGB_W, NN_W = 0.70, 0.30

preds = np.zeros(len(test))

# NN test predictions
final_nn.eval()
X_test_nn = nn_base_arr[[nn_pos.get(s, 0) for s in test["smiles"].values]]
with torch.no_grad():
    nn_test_raw = final_nn(torch.tensor(X_test_nn, dtype=torch.float32).to(DEVICE)).cpu().numpy()

for tt_idx, tt in enumerate(TARGETS):
    mask_te = (test["target_type"] == tt).values
    te_df   = test[mask_te].reset_index(drop=True)

    # LGB predictions
    Xte_lgb = make_X(te_df, tt, BEST_FEATS[tt], xf_table=wide_full)
    lgb_p   = lgb_models[tt].predict(Xte_lgb)

    # NN predictions (de-normalize)
    nn_p = (nn_test_raw[mask_te, tt_idx] * sd[tt_idx] + mu[tt_idx])

    # Blend
    blend_p = LGB_W * lgb_p + NN_W * nn_p
    preds[mask_te] = blend_p


# Apply Override: for test molecules in train with same target_type, use ground truth
override_count = 0
train_label_map = {}  # (smiles, target_type) -> mean target
for _, row in train.iterrows():
    key = (row["smiles"], row["target_type"])
    if key not in train_label_map:
        train_label_map[key] = []
    train_label_map[key].append(row["target"])
train_label_map = {k: np.mean(v) for k, v in train_label_map.items()}

for i, row in test.iterrows():
    key = (row["smiles"], row["target_type"])
    if key in train_label_map:
        preds[i] = train_label_map[key]
        override_count += 1

print(f"Override applied to {override_count} test rows ({override_count/len(test)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "target": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"  Shape: {submission.shape}")
print(f"\nSummary:")
print(f"  LGB CV: {lgb_mean_cv:.4f}")
print(f"  NN  CV: {nn_mean_cv:.4f}")
print(f"  Blend: {LGB_W}*LGB + {NN_W}*NN")
