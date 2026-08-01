# ============================================================================
#  SHARED LIBRARY  (self-contained — all helpers inlined)
# ============================================================================
import os, sys, json, time, glob, hashlib, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

TARGETS = ["egc", "egb", "ei", "eea", "eps", "nc", "tg"]
DFT_TARGETS = ["egc", "egb", "ei", "eea", "eps", "nc"]
SEED = 42

try:
    import rdkit
    _RDKIT_OK = True
except ImportError:
    print("Installing rdkit...")
    os.system(f"{sys.executable} -m pip install -q rdkit")
    import rdkit
    _RDKIT_OK = True

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Avalon import pyAvalonTools
    _HAS_AVALON = True
except ImportError:
    _HAS_AVALON = False

print(f"rdkit {rdkit.__version__} | numpy {np.__version__} | pandas {pd.__version__}")

_SKIP_DIRS = {
    "site-packages", "node_modules", ".git", ".venv", "venv", "__pycache__",
    "cache", "artifacts", ".ipynb_checkpoints", "lib", "lib64",
}

def _iter_csvs(root, max_depth=4):
    root = Path(root)
    base = len(root.resolve().parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        if len(d.resolve().parts) - base >= max_depth:
            dirnames[:] = []
        dirnames[:] = [x for x in dirnames if x not in _SKIP_DIRS and not x.startswith(".")]
        for f in filenames:
            if f.lower().endswith(".csv"):
                yield d / f

def find_data_dir():
    hits = {"train": None, "test": None, "archive_train": None, "pi1m": None}
    seen = set()
    roots = [r for r in (Path("/kaggle/input"), Path("."), Path("..")) if r.exists()]
    for root in roots:
        for p in _iter_csvs(root):
            rp = p.resolve()
            if rp in seen: continue
            seen.add(rp)
            try:
                if p.stat().st_size == 0: continue
                head = pd.read_csv(p, nrows=200)
            except Exception: continue
            cols = set(head.columns)
            if cols == {"SMILES"} or p.name.lower().startswith("pi1m"):
                hits["pi1m"] = hits["pi1m"] or p; continue
            if not {"smiles", "target_type"} <= cols: continue
            try:
                ntypes = pd.read_csv(p, usecols=["target_type"])["target_type"].nunique()
            except Exception: continue
            if "target" in cols:
                k = "train" if ntypes >= 7 else "archive_train"
            elif "id" in cols and ntypes >= 7:
                k = "test"
            else: continue
            hits[k] = hits[k] or p
    return hits

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

def mol_views(smi):
    """Return {mono, cyc, dimer} mol objects from a polymer SMILES."""
    def _mol(s):
        try: return Chem.MolFromSmiles(s)
        except: return None
    mono = _mol(smi)
    # try to close the ring by replacing * with dummy bonds
    cyc = None
    if mono is not None:
        dummies = [a.GetIdx() for a in mono.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) == 2:
            rw = Chem.RWMol(mono)
            rw.AddBond(dummies[0], dummies[1], Chem.rdchem.BondType.SINGLE)
            for idx in sorted(dummies, reverse=True):
                rw.RemoveAtom(idx)
            try:
                Chem.SanitizeMol(rw)
                cyc = rw.GetMol()
            except: pass
    # dimer
    dimer = None
    if mono is not None:
        dummies = [a.GetIdx() for a in mono.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) == 2:
            combo = Chem.CombineMols(mono, mono)
            rw = Chem.RWMol(combo)
            N = mono.GetNumAtoms()
            d1, d0b = dummies[1], dummies[0] + N
            n1 = rw.GetAtomWithIdx(d1).GetNeighbors()[0].GetIdx()
            n2 = rw.GetAtomWithIdx(d0b).GetNeighbors()[0].GetIdx()
            rw.AddBond(n1, n2, Chem.rdchem.BondType.SINGLE)
            for idx in sorted([d1, d0b], reverse=True):
                rw.RemoveAtom(idx)
            try:
                Chem.SanitizeMol(rw)
                dimer = rw.GetMol()
            except: pass
    return {"mono": mono, "cyc": cyc, "dimer": dimer}

_DESC_LIST = [d[0] for d in Descriptors._descList]

def _descriptors(mol):
    if mol is None: return np.full(len(_DESC_LIST), np.nan, dtype=np.float32)
    try:
        d = Descriptors.CalcMolDescriptors(mol)
        return np.array([d.get(k, np.nan) for k in _DESC_LIST], dtype=np.float32)
    except Exception:
        out = np.full(len(_DESC_LIST), np.nan, dtype=np.float32)
        for i, (name, fn) in enumerate(Descriptors._descList):
            try: out[i] = fn(mol)
            except: pass
        return out

def _polymer_features(views):
    mono, cyc = views["mono"], views["cyc"]
    m = cyc if cyc is not None else mono
    if m is None: return np.full(16, np.nan, dtype=np.float32)
    heavy = max(m.GetNumHeavyAtoms(), 1)
    nrot = rdMolDescriptors.CalcNumRotatableBonds(m)
    narom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    nring = rdMolDescriptors.CalcNumRings(m)
    nsp3 = sum(1 for a in m.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP3)
    mw = Descriptors.MolWt(m); tpsa = rdMolDescriptors.CalcTPSA(m); mr = Descriptors.MolMR(m)
    counts = {z: 0 for z in (9, 17, 35, 53, 14, 16, 7, 8)}
    for a in m.GetAtoms():
        z = a.GetAtomicNum()
        if z in counts: counts[z] += 1
    halo = counts[9] + counts[17] + counts[35] + counts[53]
    return np.array([heavy, nrot, nrot/heavy, narom/heavy, nring, nring/heavy, nsp3/heavy,
                     mw, mw/heavy, tpsa, tpsa/heavy, mr, mr/max(mw,1e-6),
                     halo/heavy, (counts[7]+counts[8])/heavy, counts[14]/heavy],
                    dtype=np.float32)

_POLY_NAMES = ["heavy","nrot","rot_dens","arom_frac","nring","ring_dens","sp3_frac",
               "mw","mw_per_heavy","tpsa","tpsa_dens","molmr","spec_refr",
               "halo_frac","no_frac","si_frac"]

def _fp_generators(nbits=2048):
    return {
        "mg2": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits),
        "mg3": rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=nbits),
        "ap":  rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits),
        "tt":  rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nbits),
        "rdk": rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nbits),
    }

def featurize(smiles_list, nbits=2048, want=("desc","poly","fp"), cache_tag=None, verbose=True):
    key = hashlib.md5(
        (json.dumps(list(smiles_list)) + str(nbits) + str(sorted(want))).encode()
    ).hexdigest()[:16]
    cf = CACHE / f"feat_{cache_tag or 'x'}_{key}.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        if verbose: print(f"  features from cache {cf.name}  {z['X'].shape}")
        return z["X"], list(z["cols"])
    gens = _fp_generators(nbits) if "fp" in want else {}
    rows, t0, cols = [], time.time(), None
    for i, smi in enumerate(smiles_list):
        v = mol_views(smi)
        parts, names = [], []
        if "desc" in want:
            parts.append(_descriptors(v["mono"]));  names += [f"d_mono_{n}" for n in _DESC_LIST]
            parts.append(_descriptors(v["dimer"])); names += [f"d_dim_{n}"  for n in _DESC_LIST]
        if "poly" in want:
            parts.append(_polymer_features(v));     names += [f"p_{n}" for n in _POLY_NAMES]
        if "fp" in want:
            m = v["cyc"] or v["mono"]
            for gname, g in gens.items():
                arr = (g.GetCountFingerprintAsNumPy(m).astype(np.float32)
                       if gname in ("mg2","mg3") else g.GetFingerprintAsNumPy(m).astype(np.float32))                       if m is not None else np.zeros(nbits, dtype=np.float32)
                parts.append(arr); names += [f"{gname}_{j}" for j in range(nbits)]
            mac = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32) if m is not None else np.zeros(167, dtype=np.float32)
            parts.append(mac); names += [f"maccs_{j}" for j in range(len(mac))]
            if _HAS_AVALON:
                from rdkit.Avalon import pyAvalonTools
                av = np.array(pyAvalonTools.GetAvalonFP(m, 512), dtype=np.float32) if m is not None else np.zeros(512, dtype=np.float32)
                parts.append(av); names += [f"avalon_{j}" for j in range(512)]
        rows.append(np.concatenate(parts)); cols = cols or names
        if verbose and (i+1) % 2000 == 0:
            print(f"    {i+1}/{len(smiles_list)}  {time.time()-t0:.0f}s")
    X = np.vstack(rows).astype(np.float32); X[~np.isfinite(X)] = np.nan
    np.savez_compressed(cf, X=X, cols=np.array(cols, dtype=object))
    if verbose: print(f"  featurised {X.shape} in {time.time()-t0:.0f}s -> {cf.name}")
    return X, cols

def clean_matrix(X, ref=None):
    med = np.nanmedian(ref if ref is not None else X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.clip(np.where(np.isfinite(X), X, med), -1e12, 1e12).astype(np.float32)

def make_folds(df, n_splits=10, seed=SEED):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for k, (_, va) in enumerate(skf.split(df, df["target_type"])):
        fold[va] = k
    return fold

def soft_check(cond, msg):
    if cond: return True
    print(f"\n*** CHECK FAILED: {msg}\n*** continuing\n")
    return False

def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y-p)**2); tt = np.sum((y-y.mean())**2)
    return 1.0 - ss/tt if tt > 0 else float("nan")

def score_table(df, ycol="target", pcol="pred", label=""):
    rows = []
    for t in TARGETS:
        s = df[df.target_type == t]
        rows.append({"target": t, "n": len(s), "r2": r2(s[ycol], s[pcol]) if len(s) else np.nan})
    tab = pd.DataFrame(rows); mean_r2 = tab["r2"].mean()
    print(f"\n=== {label} ===")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"MEAN R2 over 7 targets = {mean_r2:.4f}   (leaderboard ~ {100*mean_r2:.2f})")
    return mean_r2, tab

TRAIN_RANGE = {}

def set_train_range(train, *extra):
    global TRAIN_RANGE
    frames = [train[["target_type","target"]]]
    for e in extra:
        if e is not None and len(e): frames.append(e[["target_type","target"]])
    allrows = pd.concat(frames, ignore_index=True)
    TRAIN_RANGE = {t: (float(g.target.min()), float(g.target.max()))
                   for t, g in allrows.groupby("target_type")}
    return TRAIN_RANGE

def clip_to_range(df, pred_col="pred"):
    out = df[pred_col].to_numpy(float).copy()
    for t, (lo, hi) in TRAIN_RANGE.items():
        m = (df.target_type == t).to_numpy()
        pad = 0.05 * (hi - lo)
        out[m] = np.clip(out[m], lo-pad, hi+pad)
    return out

def dedupe_labels(df):
    """Keep one label per (smiles, target_type) - mean of duplicates."""
    agg = (df.groupby(["smiles","target_type"], sort=False)["target"]
             .mean().reset_index())
    print(f"  deduped {len(df)} -> {len(agg)} rows")
    return agg

def load_data():
    hits = find_data_dir()
    train = pd.read_csv(hits["train"])[["smiles","target_type","target"]]
    test  = pd.read_csv(hits["test"])[["id","smiles","target_type"]]
    arch  = pd.read_csv(hits["archive_train"])[["smiles","target_type","target"]] if hits["archive_train"] else pd.DataFrame(columns=["smiles","target_type","target"])
    pi1m  = hits["pi1m"]
    print(f"train  {train.shape}  <- {hits['train']}")
    print(f"test   {test.shape}  <- {hits['test']}")
    print(f"arch   {arch.shape}  <- {hits['archive_train']}")
    return train, test, arch, pi1m

def apply_override(test, train, arch=None, enable=True, pred_col="pred", verbose=True):
    out = test.copy()
    if not enable:
        if verbose: print("override disabled")
        return out
    frames = [train[["smiles","target_type","target"]]]
    if arch is not None and len(arch): frames.append(arch[["smiles","target_type","target"]])
    src = dedupe_labels(pd.concat(frames, ignore_index=True)).rename(columns={"target":"truth"})
    n0 = len(out)
    out = out.merge(src, on=["smiles","target_type"], how="left")
    assert len(out) == n0, "override merge duplicated rows"
    hit = out.truth.notna().to_numpy()
    out.loc[hit, pred_col] = out.loc[hit, "truth"]
    if verbose:
        print(f"\noverride: {hit.sum()}/{len(out)} test rows ({hit.mean():.1%}) replaced with measured values")
        if hit.sum(): print(out.loc[hit, "target_type"].value_counts().to_string())
    soft_check(out.loc[hit, pred_col].notna().all(), "override wrote NaN")
    return out.drop(columns=["truth"])

def write_submission(test, pred_col="pred", path=None):
    path = path or (OUT / "submission.csv")
    sub = test[["id"]].copy()
    sub["target"] = np.asarray(test[pred_col], dtype=float)
    assert sub["target"].notna().all(), "NaN in submission"
    sub = sub.sort_values("id")
    sub.to_csv(path, index=False)
    print(f"\nwrote {path}  rows={len(sub)}  ids {sub.id.min()}..{sub.id.max()}")
    print(sub.head())
    return sub

def save_oof(df, name, pred_col="pred"):
    cols = ["smiles","target_type","target",pred_col]
    o = df[cols].rename(columns={pred_col:"pred"})
    p = OUT / f"oof_{name}.csv"; o.to_csv(p, index=False)
    print(f"wrote {p}  {o.shape}"); return p

def save_pred(df, name, pred_col="pred"):
    cols = ["id","smiles","target_type",pred_col]
    o = df[cols].rename(columns={pred_col:"pred"})
    p = OUT / f"pred_{name}.csv"; o.to_csv(p, index=False)
    print(f"wrote {p}  {o.shape}"); return p

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME      = "v15_r1transfer"
N_FOLDS   = 10
NBITS     = 1024
N_SEEDS   = 3
EPOCHS    = 300
FAST      = bool(int(os.environ.get("FAST", "0")))
if FAST:
    N_SEEDS, EPOCHS = 1, 40

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} on {DEV}")

train, test, arch, _ = load_data()
train = dedupe_labels(train)
set_train_range(train, arch)
train["fold"] = make_folds(train, N_FOLDS)

# ── KEY CHANGE vs NB04: include R1 archive molecules in uniq ──────────────────
# R1-only molecules will always be in the training set (fold = -1),
# providing extra tg/egc signal for the shared NN backbone.
uniq_r2      = set(train.smiles) | set(test.smiles)
uniq_r1_only = set(arch.smiles) - uniq_r2
uniq         = sorted(uniq_r2 | uniq_r1_only)
pos          = {s: i for i, s in enumerate(uniq)}
print(f"R2 molecules : {len(uniq_r2):,}")
print(f"R1-only extra: {len(uniq_r1_only):,}")
print(f"Total uniq   : {len(uniq):,}")

X_all, cols = featurize(uniq, nbits=NBITS, want=("desc","poly","fp"), cache_tag="v15full")
cols = np.array(cols).astype(str)
dense = np.where(np.char.startswith(cols,"d_") | np.char.startswith(cols,"p_"))[0]
_d = set(dense.tolist())
fp = np.array([i for i in range(len(cols)) if i not in _d])
on = (X_all[:, fp] > 0).mean(0)
keep = fp[(on > 0.005) & (on < 0.995)]

Xd = clean_matrix(X_all[:, dense])
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=SEED)
Xd = qt.fit_transform(Xd).astype(np.float32)
Xf = np.minimum(X_all[:, keep], 4.0).astype(np.float32)
X  = np.hstack([Xd, Xf]).astype(np.float32)
print(f"input matrix {X.shape}  (dense {Xd.shape[1]} + fp {Xf.shape[1]})")

n_mol = len(uniq)
Y  = np.full((n_mol, len(TARGETS)), np.nan, dtype=np.float32)
Fk = np.full((n_mol, len(TARGETS)), -1,    dtype=np.int64)

# R2 training labels (with proper fold assignments)
for s, t, v, k in zip(train.smiles, train.target_type, train.target, train.fold):
    j = TARGETS.index(t)
    Y[pos[s], j] = v
    Fk[pos[s], j] = k

# R1 archive labels — tg and egc only, fold=-1 (always in training, never held out)
# This gives the NN ~6k extra labeled rows, enriching the shared backbone.
ARCH_TARGETS = {"tg", "egc"}
arch_added = 0
for s, t, v in zip(arch.smiles, arch.target_type, arch.target):
    if t not in ARCH_TARGETS: continue
    j = TARGETS.index(t)
    i = pos[s]
    if np.isnan(Y[i, j]):      # don't overwrite R2 labels
        Y[i, j] = v
        # Fk[i, j] stays -1 → always in training
        arch_added += 1

KNOWN = np.isfinite(Y)
mu = np.array([np.nanmean(Y[:, j]) for j in range(len(TARGETS))], dtype=np.float32)
sd = np.array([np.nanstd(Y[:, j]) + 1e-9 for j in range(len(TARGETS))], dtype=np.float32)
Yz = np.where(KNOWN, (Y-mu)/sd, 0.0).astype(np.float32)

print("Known cells per target:", dict(zip(TARGETS, KNOWN.sum(0).tolist())))
print(f"R1 archive cells added: {arch_added}  (tg+egc extra labels)")
print(f"Fold-safety: Fk==-1 cells (R1-only, always train): {(Fk==-1).sum()}")

class MultiTaskNet(nn.Module):
    def __init__(self, d_in, n_out, trunk=(1024, 512, 256), head=128, p=0.25):
        super().__init__()
        layers, d = [], d_in
        for h in trunk:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d,head), nn.SiLU(), nn.Dropout(p/2), nn.Linear(head,1))
            for _ in range(n_out)
        )

    def forward(self, x):
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=1)

def masked_huber(pred, y, mask):
    """Huber loss over observed cells only, averaged per target then summed."""
    loss = F.huber_loss(pred, y, reduction="none", delta=1.0) * mask
    per_t = loss.sum(0) / mask.sum(0).clamp(min=1.0)
    return per_t[mask.sum(0) > 0].mean()

def fit_one(train_mask, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    xt = torch.tensor(X,  device=DEV)
    yt = torch.tensor(Yz, device=DEV)
    mt = torch.tensor(train_mask.astype(np.float32), device=DEV)
    rows  = np.where(train_mask.any(1))[0]
    model = MultiTaskNet(X.shape[1], len(TARGETS)).to(DEV)
    opt   = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3,
        total_steps=EPOCHS * max(1, len(rows)//256 + 1), pct_start=0.15
    )
    bs = 256
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(rows)
        for i in range(0, len(perm), bs):
            b = torch.tensor(perm[i:i+bs], device=DEV)
            opt.zero_grad(set_to_none=True)
            loss = masked_huber(model(xt[b]), yt[b], mt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, n_mol, 4096):
            out.append(model(xt[i:i+4096]).cpu().numpy())
    return np.vstack(out) * sd + mu

t0         = time.time()
folds_run  = 3 if FAST else N_FOLDS
oof_mat    = np.full((n_mol, len(TARGETS)), np.nan, dtype=np.float32)

for k in range(folds_run):
    # training mask: known cells that are NOT in R2-fold-k
    # R1-only cells (Fk==-1) satisfy Fk!=k for all k=0..9, so always included
    tm  = KNOWN & (Fk != k)
    acc = np.zeros((n_mol, len(TARGETS)), dtype=np.float64)
    for s in range(N_SEEDS):
        acc += fit_one(tm, SEED + 100*s + k)
    p = acc / N_SEEDS
    # held-out: only R2 cells in this fold
    held = KNOWN & (Fk == k)
    oof_mat[held] = p[held].astype(np.float32)
    print(f"  fold {k}  held {held.sum():>5} R2-cells   {time.time()-t0:.0f}s")

# Full-data model for test predictions
acc = np.zeros((n_mol, len(TARGETS)), dtype=np.float64)
for s in range(N_SEEDS):
    acc += fit_one(KNOWN, SEED + 7*s)
full_mat = acc / N_SEEDS
print(f"total {time.time()-t0:.0f}s")

# stageA: full-data predictions everywhere, OOF overrides for R2 train molecules
stageA = pd.DataFrame(full_mat, index=uniq, columns=TARGETS)
for j, t in enumerate(TARGETS):
    m = KNOWN[:, j] & np.isfinite(oof_mat[:, j]) & (Fk[:, j] >= 0)  # R2 train only
    stageA.iloc[np.where(m)[0], j] = oof_mat[m, j]
stageA.index.name = "smiles"
stageA.to_csv(OUT / f"stageA_{NAME}.csv")

# Score NN OOF on R2 training rows
train["pred"] = [oof_mat[pos[s], TARGETS.index(t)] for s, t in zip(train.smiles, train.target_type)]
test["pred"]  = [full_mat[pos[s], TARGETS.index(t)] for s, t in zip(test.smiles, test.target_type)]
test["pred"]  = clip_to_range(test)
score_table(train, label="Stage A NN OOF (R2 train)")

import scipy.optimize as sopt
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb

N_AUG = 8    # masked sibling replicas (more = more robust to missing siblings)
SIB   = DFT_TARGETS + ["tg"]

def label_table(df):
    d = dedupe_labels(df[["smiles","target_type","target"]])
    return d.pivot(index="smiles", columns="target_type", values="target")              .reindex(columns=TARGETS).reindex(uniq)

fold_tables = {k: label_table(train[train.fold != k]) for k in range(N_FOLDS)}
full_table  = label_table(pd.concat([
    train[["smiles","target_type","target"]],
    arch[["smiles","target_type","target"]]
], ignore_index=True))
print("sibling table (predict time):", int(full_table.notna().sum().sum()), "known cells")

_stageA_np = stageA.to_numpy()   # positional access via pos[]

def make_features(smiles, target, table, rng=None, mask_p=0.0):
    idx = np.array([pos[s] for s in smiles])
    A = _stageA_np[idx]                              # (n, 7)
    T = table.to_numpy()[idx].astype(float)          # (n, 7) true siblings
    tcol = TARGETS.index(target)
    T = T.copy(); T[:, tcol] = np.nan               # never see own answer
    known = np.isfinite(T)
    if mask_p > 0 and rng is not None:
        known = known & (rng.random(known.shape) > mask_p)
    T = np.where(known, T, np.nan)
    filled = np.where(known, T, A)
    resid  = np.where(known, T - A, 0.0)
    sib_cols = [i for i, x in enumerate(TARGETS) if x != target]
    f = {}
    for i, name in enumerate(TARGETS):         f[f"A_{name}"]    = A[:, i]
    for i in sib_cols:
        f[f"fill_{TARGETS[i]}"]  = filled[:, i]
        f[f"known_{TARGETS[i]}"] = known[:, i].astype(float)
        f[f"res_{TARGETS[i]}"]   = resid[:, i]
    f["n_known"] = known[:, sib_cols].sum(1).astype(float)
    g = {k: filled[:, TARGETS.index(k)] for k in TARGETS}
    f["phys_gap"]    = g["ei"] - g["eea"]
    f["phys_nc2"]    = g["nc"] ** 2
    f["phys_ll"]     = (g["nc"]**2 - 1) / (g["nc"]**2 + 2)
    f["phys_eps_ll"] = (g["eps"] - 1)   / (g["eps"] + 2)
    f["phys_dgap"]   = g["egb"] - g["egc"]
    f["phys_mid"]    = 0.5 * (g["ei"] + g["eea"])
    X = pd.DataFrame(f, index=range(len(smiles)))
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

def stage_b_models(n):
    small = n < 600
    return {
        "lgb": lambda: lgb.LGBMRegressor(
            n_estimators=120 if FAST else (500 if small else 900),
            learning_rate=0.03, num_leaves=7 if small else 31,
            min_child_samples=5 if small else 20, colsample_bytree=0.7,
            subsample=0.8, subsample_freq=1, reg_lambda=3.0 if small else 1.0,
            verbose=-1, n_jobs=-1, random_state=SEED),
        "ridge": lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30))),
        "et": lambda: ExtraTreesRegressor(
            n_estimators=400, max_features=0.5, min_samples_leaf=2,
            n_jobs=-1, random_state=SEED),
    }

train["pred"] = np.nan
test["pred"]  = np.nan
rng    = np.random.default_rng(SEED)
nf     = 3 if FAST else N_FOLDS
report = []

for t in TARGETS:
    mtr = (train.target_type == t).to_numpy()
    sub = train.loc[mtr].reset_index(drop=True)
    y   = sub.target.to_numpy(float)
    fold = sub.fold.to_numpy()
    zoo  = stage_b_models(len(y))
    mu_t, sd_t = y.mean(), y.std() + 1e-9

    oof = {m: np.full(len(y), np.nan) for m in zoo}
    for k in range(nf):
        a, b = np.where(fold != k)[0], np.where(fold == k)[0]
        if not len(b): continue
        tbl = fold_tables[k]
        Xs, ys = [], []
        for r in range(1 + (0 if FAST else N_AUG)):
            Xs.append(make_features(sub.smiles.values[a], t, tbl,
                                    rng=rng, mask_p=0.0 if r==0 else 0.35))
            ys.append(y[a])
        Xa, ya = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
        Xb = make_features(sub.smiles.values[b], t, tbl)
        for m, build in zoo.items():
            oof[m][b] = build().fit(Xa, (ya-mu_t)/sd_t).predict(Xb) * sd_t + mu_t

    P = np.column_stack([np.where(np.isnan(oof[m]), mu_t, oof[m]) for m in zoo])
    w, _ = sopt.nnls(P, y)
    w = w/w.sum() if w.sum() > 1e-9 else np.full(P.shape[1], 1/P.shape[1])
    train.loc[mtr, "pred"] = P @ w

    Xs, ys = [], []
    for r in range(1 + (0 if FAST else N_AUG)):
        Xs.append(make_features(sub.smiles.values, t, full_table,
                                rng=rng, mask_p=0.0 if r==0 else 0.35))
        ys.append(y)
    Xa, ya = pd.concat(Xs, ignore_index=True), np.concatenate(ys)
    mte = (test.target_type == t).to_numpy()
    Xb  = make_features(test.loc[mte, "smiles"].values, t, full_table)
    Pt  = np.column_stack([build().fit(Xa, (ya-mu_t)/sd_t).predict(Xb) * sd_t + mu_t
                           for build in zoo.values()])
    test.loc[mte, "pred"] = Pt @ w

    base = r2(y, stageA.loc[sub.smiles.values, t].to_numpy())
    stgb = r2(y, P@w)
    report.append({"target":t,"n":len(y),"stageA_r2":base,"stageB_r2":stgb,"gain":stgb-base})
    print(f"{t:>4}  StageA {base:.4f} -> StageB {stgb:.4f}  ({stgb-base:+.4f})"
          f"   w={dict(zip(zoo, np.round(w,2)))}")

print()
print(pd.DataFrame(report).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# Fold-safety check
_probe = make_features(train.smiles.values[:50], "eps", full_table)
assert not any(c.endswith("_eps") and c.startswith(("fill_","known_","res_"))
               for c in _probe.columns), "target leaked into sibling block"
for k in range(min(3, N_FOLDS)):
    held = train[train.fold == k]
    tbl  = fold_tables[k]
    bad  = sum(1 for s, tt in zip(held.smiles, held.target_type)
               if s in tbl.index and np.isfinite(tbl.at[s, tt])
               and not ((train.fold != k) & (train.smiles == s) & (train.target_type == tt)).any())
    assert bad == 0, f"fold {k}: {bad} held-out labels visible in sibling table"
print("fold-safety assertions passed")

train["pred"] = clip_to_range(train)
test["pred"]  = clip_to_range(test)
mean_r2, tab  = score_table(train, label="v15 sibling stacking (OOF)")
save_oof(train, NAME)
save_pred(test, NAME)

sub_df = apply_override(test, train, arch, enable=True)
write_submission(sub_df)