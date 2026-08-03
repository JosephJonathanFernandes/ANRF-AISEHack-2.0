"""
AISEHack 2.0 Round 2 - AttentiveFP multi-task GNN
Run locally (needs a GPU to be fast; falls back to CPU automatically).

Requirements:
    pip install torch torch_geometric rdkit
    (If torch_geometric's optional extensions fail to build, note that
    AttentiveFP itself only needs core torch_geometric >=2.x, no
    torch-scatter/torch-sparse required.)

Design notes, since you can't see my reasoning otherwise:
- ONE shared AttentiveFP encoder with 7 output heads (out_channels=7),
  trained jointly on all targets with a masked loss (each row only
  contributes gradient to its own target's output column). This lets
  the data-rich targets (tg, egc) help the data-poor ones (eps, nc, ei,
  eea, egb) via the shared graph representation - the GNN analogue of
  the cross-property trick from the LightGBM pipeline.
- Targets are z-scored per-target before training (their raw scales
  differ by ~500x: tg std~109 vs nc std~0.24) so the shared loss
  doesn't get dominated by tg. De-normalized before scoring/submission.
- GroupKFold(5) by SMILES for CV, same as the LightGBM pipeline, so
  the two CV numbers are directly comparable.
- Deliberately does NOT include the cross-property numeric features
  the LightGBM model uses - keeping this a clean, structurally
  different model makes it a better ensemble partner. Blending this
  with the LightGBM submission is a natural next step once you have
  both sets of predictions.
- I could not test-run this end to end (no GPU / no disk space left
  in this sandbox for a torch install) - the featurization function
  below WAS verified against all 10,605 real train+test SMILES with
  zero failures, but the model/training code is unexecuted. Sanity
  check with EPOCHS=2 before committing to a full run.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

RDLogger.DisableLog('rdApp.*')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

TARGETS = ['egc', 'egb', 'ei', 'eea', 'eps', 'nc', 'tg']
TARGET_IDX = {t: i for i, t in enumerate(TARGETS)}
EPOCHS = 200  # set to 2 for a quick sanity check that everything runs end to end

train = pd.read_csv('train.csv')   # adjust paths to wherever you keep the data locally
test = pd.read_csv('test.csv')

# ---------------------------------------------------------------
# Featurization (verified against all real SMILES - see docstring)
# ---------------------------------------------------------------
ATOM_VOCAB = ['C', 'O', 'N', '*', 'F', 'S', 'Si', 'Cl', 'P', 'Br', 'I', 'H', 'other']
DEGREE_VOCAB = [0, 1, 2, 3, 4, 'other']
HYBRID_VOCAB = ['SP', 'SP2', 'SP3', 'S', 'SP3D', 'SP3D2', 'UNSPECIFIED', 'other']
HCOUNT_VOCAB = [0, 1, 2, 3, 4, 'other']
BOND_VOCAB = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
ATOM_DIM = len(ATOM_VOCAB) + len(DEGREE_VOCAB) + len(HYBRID_VOCAB) + len(HCOUNT_VOCAB) + 3
BOND_DIM = len(BOND_VOCAB) + 2

def onehot(val, vocab):
    v = [0] * len(vocab)
    v[vocab.index(val) if val in vocab else len(vocab) - 1] = 1
    return v

def atom_features(atom):
    return (onehot(atom.GetSymbol(), ATOM_VOCAB)
            + onehot(atom.GetDegree(), DEGREE_VOCAB)
            + onehot(str(atom.GetHybridization()), HYBRID_VOCAB)
            + onehot(atom.GetTotalNumHs(), HCOUNT_VOCAB)
            + [atom.GetFormalCharge(), int(atom.GetIsAromatic()), int(atom.IsInRing())])

def bond_features(bond):
    return (onehot(bond.GetBondType(), BOND_VOCAB)
            + [int(bond.GetIsConjugated()), int(bond.IsInRing())])

def smiles_to_data(smi):
    mol = Chem.MolFromSmiles(smi)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, edge_feats = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]; dst += [j, i]; edge_feats += [f, f]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Build each unique molecule's graph once, reuse across rows
all_smiles = pd.unique(pd.concat([train['smiles'], test['smiles']], ignore_index=True))
graph_cache = {s: smiles_to_data(s) for s in all_smiles}
print(f'Built {len(graph_cache)} molecular graphs, atom_dim={ATOM_DIM}, bond_dim={BOND_DIM}')

def make_dataset(df, target_mean=None, target_std=None, has_labels=True):
    ds = []
    for row in df.itertuples():
        g = graph_cache[row.smiles]
        d = Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)
        t = TARGET_IDX[row.target_type]
        d.task = torch.tensor([t], dtype=torch.long)
        if has_labels:
            y = (row.target - target_mean[t]) / target_std[t]
            d.y = torch.tensor([y], dtype=torch.float)
        else:
            d.row_id = torch.tensor([row.id], dtype=torch.long)
        ds.append(d)
    return ds

def make_model():
    return AttentiveFP(in_channels=ATOM_DIM, hidden_channels=128, out_channels=len(TARGETS),
                        edge_dim=BOND_DIM, num_layers=2, num_timesteps=2, dropout=0.2).to(device)

def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = out.gather(1, batch.task.view(-1, 1)).squeeze(1)
            loss = F.mse_loss(pred, batch.y)
            if train_mode:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
    return total_loss / n

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, tasks = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = out.gather(1, batch.task.view(-1, 1)).squeeze(1)
        preds.append(pred.cpu().numpy())
        tasks.append(batch.task.cpu().numpy().ravel())
    return np.concatenate(preds), np.concatenate(tasks)

def train_model(train_ds, val_ds, epochs=EPOCHS, patience=20, batch_size=64, lr=1e-3):
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    best_loss, best_state, bad_epochs = float('inf'), None, 0
    for epoch in range(epochs):
        tr_loss = run_epoch(model, train_loader, opt)
        va_loss = run_epoch(model, val_loader, None)
        sched.step(va_loss)
        if va_loss < best_loss - 1e-5:
            best_loss, best_state, bad_epochs = va_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if epoch % 10 == 0:
            print(f'  epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}')
        if bad_epochs >= patience:
            print(f'  early stop at epoch {epoch}')
            break
    model.load_state_dict(best_state)
    return model

# ---------------------------------------------------------------
# GroupKFold CV (5 folds, grouped by SMILES - comparable to the LightGBM CV)
# ---------------------------------------------------------------
gkf = GroupKFold(n_splits=5)
oof_pred = np.zeros(len(train))
for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train['smiles'])):
    tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
    t_mean = tr_df.groupby('target_type')['target'].mean().reindex(TARGETS).values
    t_std = tr_df.groupby('target_type')['target'].std().reindex(TARGETS).values
    tr_ds = make_dataset(tr_df, t_mean, t_std)
    va_ds = make_dataset(va_df, t_mean, t_std)
    print(f'Fold {fold}: train={len(tr_ds)} val={len(va_ds)}')
    model = train_model(tr_ds, va_ds)
    pred_norm, tasks = predict(model, DataLoader(va_ds, batch_size=256, shuffle=False))
    pred = pred_norm * t_std[tasks] + t_mean[tasks]  # de-normalize
    oof_pred[va_idx] = pred

print(f'\n{"target":6s} {"n":>5s} {"R2":>8s}')
per_target = {}
for tt in TARGETS:
    mask = (train['target_type'] == tt).values
    per_target[tt] = r2_score(train['target'].values[mask], oof_pred[mask])
    print(f'{tt:6s} {mask.sum():5d} {per_target[tt]:8.4f}')
print(f'{"MEAN":6s} {"":5s} {np.mean(list(per_target.values())):8.4f}')

# ---------------------------------------------------------------
# Final refit on full train, predict test
# ---------------------------------------------------------------
t_mean_full = train.groupby('target_type')['target'].mean().reindex(TARGETS).values
t_std_full = train.groupby('target_type')['target'].std().reindex(TARGETS).values
full_ds = make_dataset(train, t_mean_full, t_std_full)
rng = np.random.RandomState(42)
perm = rng.permutation(len(full_ds))
full_ds = [full_ds[i] for i in perm]
# small held-out slice just to drive early stopping for the final fit
n_val = max(200, len(full_ds) // 10)
final_train_ds, final_val_ds = full_ds[n_val:], full_ds[:n_val]
final_model = train_model(final_train_ds, final_val_ds)

test_ds = make_dataset(test, has_labels=False)
pred_norm, tasks = predict(final_model, DataLoader(test_ds, batch_size=256, shuffle=False))
pred = pred_norm * t_std_full[tasks] + t_mean_full[tasks]
row_ids = np.concatenate([d.row_id.numpy() for d in test_ds])

submission = pd.DataFrame({'id': row_ids, 'target': pred}).sort_values('id')
submission.to_csv('submission_gnn.csv', index=False)
print('\nSaved submission_gnn.csv:', submission.shape)
