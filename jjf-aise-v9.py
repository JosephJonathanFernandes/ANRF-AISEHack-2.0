# !pip install rdkit torch torch_geometric transformers -q

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
print('Base imports successful.')

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP

from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Torch device: {DEVICE}')

# -----------------------------------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------------------------------
train_path = 'dataset/train.csv'
test_path  = 'dataset/test.csv'

# Adjust paths if we are inside a subfolder
if not os.path.exists(train_path):
    train_path = '../dataset/train.csv'
    test_path  = '../dataset/test.csv'

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

train_tg  = train[train['target_type'] == 'tg'].reset_index(drop=True)
train_egc = train[train['target_type'] == 'egc'].reset_index(drop=True)
test_tg   = test[test['target_type'] == 'tg'].reset_index(drop=True)
test_egc  = test[test['target_type'] == 'egc'].reset_index(drop=True)

y_tg  = train_tg['target'].values
y_egc = train_egc['target'].values

print(f'Tg  train: {len(train_tg):,}  test: {len(test_tg):,}')
print(f'Egc train: {len(train_egc):,}  test: {len(test_egc):,}')
print(f'Tg  range: [{y_tg.min():.1f}, {y_tg.max():.1f}]')
print(f'Egc range: [{y_egc.min():.4f}, {y_egc.max():.4f}]')

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (RDKit, Morgan, MACCS)
# -----------------------------------------------------------------------------
DESC_NAMES  = [n for n, _ in Descriptors.descList]
MORGAN_ECFP4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MORGAN_ECFP6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
RDKIT_FPGEN  = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)

def featurize(smiles_list):
    rdkit_rows, ecfp4_rows, ecfp6_rows, rdk_rows, maccs_rows = [], [], [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rdkit_rows.append([np.nan] * len(DESC_NAMES))
            ecfp4_rows.append(np.zeros(2048, dtype=np.uint8))
            ecfp6_rows.append(np.zeros(2048, dtype=np.uint8))
            rdk_rows.append(np.zeros(2048, dtype=np.uint8))
            maccs_rows.append(np.zeros(167, dtype=np.uint8))
        else:
            vals = Descriptors.CalcMolDescriptors(mol)
            rdkit_rows.append(list(vals.values()))
            ecfp4_rows.append(MORGAN_ECFP4.GetFingerprintAsNumPy(mol))
            ecfp6_rows.append(MORGAN_ECFP6.GetFingerprintAsNumPy(mol))
            rdk_rows.append(RDKIT_FPGEN.GetFingerprintAsNumPy(mol))
            fp_mac = MACCSkeys.GenMACCSKeys(mol)
            maccs_rows.append(np.array(fp_mac, dtype=np.uint8))

    return pd.concat([
        pd.DataFrame(rdkit_rows,  columns=DESC_NAMES),
        pd.DataFrame(ecfp4_rows,  columns=[f'ecfp4_{i}'  for i in range(2048)]),
        pd.DataFrame(ecfp6_rows,  columns=[f'ecfp6_{i}'  for i in range(2048)]),
        pd.DataFrame(rdk_rows,    columns=[f'rdkfp_{i}'  for i in range(2048)]),
        pd.DataFrame(maccs_rows,  columns=[f'maccs_{i}'  for i in range(167)]),
    ], axis=1)

def build_preprocessor(X_raw):
    X = X_raw.replace([np.inf, -np.inf], np.nan).astype(float)
    X = X.dropna(axis=1, thresh=int(0.2 * len(X)))
    X = X.loc[:, X.var() > 0]
    good_cols = X.columns.tolist()
    imputer  = SimpleImputer(strategy='median')
    X_imp    = imputer.fit_transform(X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, (imputer, scaler, good_cols)

def apply_preprocessor(X_raw, preprocessor):
    imputer, scaler, good_cols = preprocessor
    X = X_raw.replace([np.inf, -np.inf], np.nan).astype(float)
    X = X.reindex(columns=good_cols, fill_value=np.nan)
    return scaler.transform(imputer.transform(X))

print('Feature functions defined.')

print('Featurizing Tg train  ...', flush=True)
X_tg_raw       = featurize(train_tg['smiles'].tolist())
print('Featurizing Egc train ...', flush=True)
X_egc_raw      = featurize(train_egc['smiles'].tolist())
print('Featurizing Tg test   ...', flush=True)
X_tg_test_raw  = featurize(test_tg['smiles'].tolist())
print('Featurizing Egc test  ...', flush=True)
X_egc_test_raw = featurize(test_egc['smiles'].tolist())

print(f'\nRaw feature shape: {X_tg_raw.shape}')
print('Preprocessing tabular features...')

X_tg,  tg_prep  = build_preprocessor(X_tg_raw)
X_egc, egc_prep = build_preprocessor(X_egc_raw)
X_tg_test  = apply_preprocessor(X_tg_test_raw,  tg_prep)
X_egc_test = apply_preprocessor(X_egc_test_raw, egc_prep)

print(f'Tg  features after cleaning: {X_tg.shape[1]:,}')
print(f'Egc features after cleaning: {X_egc.shape[1]:,}')


# -----------------------------------------------------------------------------
# 3. CHEMBERTA-2 EMBEDDINGS (NEW FOR V9)
# -----------------------------------------------------------------------------
print('\nLoading ChemBERTa-2 model...')
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chemberta = AutoModel.from_pretrained(model_name).to(DEVICE)
chemberta.eval()

def get_chemberta_embeddings(smiles_list, batch_size=256):
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        inputs = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors='pt', max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = chemberta(**inputs)
            # Use CLS token representation
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

print('Extracting ChemBERTa embeddings for Tg train...')
emb_tg = get_chemberta_embeddings(train_tg['smiles'].tolist())
print('Extracting ChemBERTa embeddings for Egc train...')
emb_egc = get_chemberta_embeddings(train_egc['smiles'].tolist())
print('Extracting ChemBERTa embeddings for Tg test...')
emb_tg_test = get_chemberta_embeddings(test_tg['smiles'].tolist())
print('Extracting ChemBERTa embeddings for Egc test...')
emb_egc_test = get_chemberta_embeddings(test_egc['smiles'].tolist())

print(f'ChemBERTa embeddings shape: {emb_tg.shape}')


# -----------------------------------------------------------------------------
# 4. GRAPH NEURAL NETWORK (AttentiveFP)
# -----------------------------------------------------------------------------
ATOM_LIST = ['C','N','O','S','F','Si','P','Cl','Br','I','B','*','Other']

def one_hot(val, choices):
    vec = [0] * len(choices)
    idx = choices.index(val) if val in choices else len(choices) - 1
    vec[idx] = 1
    return vec

def atom_features(atom):
    feats = one_hot(atom.GetSymbol(), ATOM_LIST)
    feats += [
        atom.GetDegree(), atom.GetFormalCharge(), int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(), int(atom.GetHybridization()),
    ]
    return feats

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

def smiles_to_graph(smiles, y=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]

    if len(edge_index) == 0:          # single-atom edge case
        edge_index = [[0, 0]]
        edge_attr  = [[0] * 6]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data

NODE_DIM = len(ATOM_LIST) + 5
EDGE_DIM = 6
print(f'NODE_DIM={NODE_DIM}  EDGE_DIM={EDGE_DIM}')

# Target scaling helps GNN training stability
y_tg_scaler  = StandardScaler()
y_egc_scaler = StandardScaler()
y_tg_scaled  = y_tg_scaler.fit_transform(y_tg.reshape(-1, 1)).ravel()
y_egc_scaled = y_egc_scaler.fit_transform(y_egc.reshape(-1, 1)).ravel()

print('Building Tg graphs ...')
graphs_tg = [smiles_to_graph(s, y) for s, y in zip(train_tg['smiles'], y_tg_scaled)]
graphs_tg_test = [smiles_to_graph(s) for s in test_tg['smiles']]

print('Building Egc graphs ...')
graphs_egc = [smiles_to_graph(s, y) for s, y in zip(train_egc['smiles'], y_egc_scaled)]
graphs_egc_test = [smiles_to_graph(s) for s in test_egc['smiles']]


GNN_SEEDS = [42, 7, 123]   # 3 seeds for GNN

def train_gnn_ensemble(graphs, y_raw, graphs_test, scaler, seed, n_splits=5,
                       hidden=64, layers=2, timesteps=2, epochs=60, patience=12, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(graphs))
    oof_scaled  = np.zeros(len(graphs))
    test_scaled = np.zeros(len(graphs_test))

    test_loader = DataLoader(graphs_test, batch_size=256, shuffle=False)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(idx), 1):
        train_loader = DataLoader([graphs[i] for i in tr_idx], batch_size=64, shuffle=True)
        val_loader   = DataLoader([graphs[i] for i in val_idx], batch_size=128, shuffle=False)

        model = AttentiveFP(
            in_channels=NODE_DIM, hidden_channels=hidden, out_channels=1,
            edge_dim=EDGE_DIM, num_layers=layers, num_timesteps=timesteps, dropout=0.1
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        best_val_loss, bad_epochs, best_state = float('inf'), 0, None

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = loss_fn(out.squeeze(-1), batch.y)
                loss.backward()
                opt.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    val_losses.append(loss_fn(out.squeeze(-1), batch.y).item())
            val_loss = np.mean(val_losses)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                val_preds.append(out.squeeze(-1).cpu().numpy())
        oof_scaled[val_idx] = np.concatenate(val_preds)

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                test_preds.append(out.squeeze(-1).cpu().numpy())
        test_scaled += np.concatenate(test_preds) / n_splits

        r2_fold = r2_score(y_raw[val_idx], scaler.inverse_transform(
            oof_scaled[val_idx].reshape(-1, 1)).ravel())
        print(f'    fold {fold} | epochs={epoch+1:2d}  GNN R²={r2_fold:.4f}')

    oof  = scaler.inverse_transform(oof_scaled.reshape(-1, 1)).ravel()
    test = scaler.inverse_transform(test_scaled.reshape(-1, 1)).ravel()
    print(f'  OOF GNN R²: {r2_score(y_raw, oof):.4f}')
    return oof, test

print('=' * 60)
print('  GNN — Tg')
print('=' * 60)

gnn_oof_tg_all, gnn_test_tg_all = [], []
for seed in GNN_SEEDS:
    print(f'\n  seed={seed}')
    oof_g, test_g = train_gnn_ensemble(graphs_tg, y_tg, graphs_tg_test, y_tg_scaler, seed=seed)
    gnn_oof_tg_all.append(oof_g)
    gnn_test_tg_all.append(test_g)

oof_g_tg  = np.mean(gnn_oof_tg_all,  axis=0)
test_g_tg = np.mean(gnn_test_tg_all, axis=0)
print(f'\nSeed-averaged GNN OOF R² (Tg): {r2_score(y_tg, oof_g_tg):.4f}')

print('=' * 60)
print('  GNN — Egc')
print('=' * 60)

gnn_oof_egc_all, gnn_test_egc_all = [], []
for seed in GNN_SEEDS:
    print(f'\n  seed={seed}')
    oof_g, test_g = train_gnn_ensemble(graphs_egc, y_egc, graphs_egc_test, y_egc_scaler, seed=seed)
    gnn_oof_egc_all.append(oof_g)
    gnn_test_egc_all.append(test_g)

oof_g_egc  = np.mean(gnn_oof_egc_all,  axis=0)
test_g_egc = np.mean(gnn_test_egc_all, axis=0)
print(f'\nSeed-averaged GNN OOF R² (Egc): {r2_score(y_egc, oof_g_egc):.4f}')


# -----------------------------------------------------------------------------
# 5. TABULAR MODELS (LGBM + XGB) & CHEMBERTA RIDGE
# -----------------------------------------------------------------------------
SEEDS = [42, 7, 123, 17, 99]

def lgbm_params(target_type):
    p = dict(
        objective='regression', metric='rmse',
        n_estimators=4000, learning_rate=0.01,
        num_leaves=127, max_depth=-1,
        min_child_samples=15,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.4,
        reg_alpha=0.05, reg_lambda=1.0,
        n_jobs=-1, verbose=-1,
    )
    if target_type == 'egc':
        p['num_leaves'] = 63
        p['min_child_samples'] = 20
    return p

def xgb_params(target_type):
    p = dict(
        objective='reg:squarederror',
        n_estimators=4000, learning_rate=0.01,
        max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.4,
        reg_alpha=0.05, reg_lambda=1.0,
        n_jobs=-1, tree_method='hist',
        early_stopping_rounds=200,
    )
    if target_type == 'egc':
        p['max_depth'] = 5
    return p

def train_ensemble(X_train, y_train, X_test, target_type, seed, n_splits=10):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test  = np.asarray(X_test)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_lgbm  = np.zeros(len(X_train))
    oof_xgb   = np.zeros(len(X_train))
    test_lgbm = np.zeros(len(X_test))
    test_xgb  = np.zeros(len(X_test))

    lp = lgbm_params(target_type); lp['random_state'] = seed
    xp = xgb_params(target_type);  xp['random_state'] = seed

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        m_lgbm = lgb.LGBMRegressor(**lp)
        m_lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(200, verbose=False),
                              lgb.log_evaluation(period=0)])
        oof_lgbm[val_idx] = m_lgbm.predict(X_val)
        test_lgbm        += m_lgbm.predict(X_test) / n_splits

        m_xgb = xgb.XGBRegressor(**xp)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = m_xgb.predict(X_val)
        test_xgb        += m_xgb.predict(X_test) / n_splits

    return (oof_lgbm, oof_xgb), (test_lgbm, test_xgb)

def train_ridge(X_train, y_train, X_test, seed, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train))
    test = np.zeros(len(X_test))
    
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        # Simple Ridge on embeddings
        model = Ridge(alpha=10.0, random_state=seed)
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        test += model.predict(X_test) / n_splits
        
    return oof, test

print('=' * 60)
print('  Tabular (LGBM+XGB) & ChemBERTa Ridge — Tg')
print('=' * 60)

tg_oof_parts_all, tg_test_parts_all = [], []
tg_oof_ridge_all, tg_test_ridge_all = [], []

for seed in SEEDS:
    print(f'\n  seed={seed}')
    oof_parts, test_parts = train_ensemble(X_tg, y_tg, X_tg_test, 'tg', seed=seed)
    tg_oof_parts_all.append(oof_parts)
    tg_test_parts_all.append(test_parts)
    
    oof_r, test_r = train_ridge(emb_tg, y_tg, emb_tg_test, seed=seed)
    tg_oof_ridge_all.append(oof_r)
    tg_test_ridge_all.append(test_r)

oof_l_tg  = np.mean([p[0] for p in tg_oof_parts_all],  axis=0)
oof_x_tg  = np.mean([p[1] for p in tg_oof_parts_all],  axis=0)
test_l_tg = np.mean([p[0] for p in tg_test_parts_all], axis=0)
test_x_tg = np.mean([p[1] for p in tg_test_parts_all], axis=0)

oof_r_tg  = np.mean(tg_oof_ridge_all, axis=0)
test_r_tg = np.mean(tg_test_ridge_all, axis=0)

print(f'\nSeed-averaged OOF R²  LGBM={r2_score(y_tg, oof_l_tg):.4f}  '
      f'XGB={r2_score(y_tg, oof_x_tg):.4f}  ChemBERTa-Ridge={r2_score(y_tg, oof_r_tg):.4f}')


print('=' * 60)
print('  Tabular (LGBM+XGB) & ChemBERTa Ridge — Egc')
print('=' * 60)

egc_oof_parts_all, egc_test_parts_all = [], []
egc_oof_ridge_all, egc_test_ridge_all = [], []

for seed in SEEDS:
    print(f'\n  seed={seed}')
    oof_parts, test_parts = train_ensemble(X_egc, y_egc, X_egc_test, 'egc', seed=seed)
    egc_oof_parts_all.append(oof_parts)
    egc_test_parts_all.append(test_parts)

    oof_r, test_r = train_ridge(emb_egc, y_egc, emb_egc_test, seed=seed)
    egc_oof_ridge_all.append(oof_r)
    egc_test_ridge_all.append(test_r)

oof_l_egc  = np.mean([p[0] for p in egc_oof_parts_all],  axis=0)
oof_x_egc  = np.mean([p[1] for p in egc_oof_parts_all],  axis=0)
test_l_egc = np.mean([p[0] for p in egc_test_parts_all], axis=0)
test_x_egc = np.mean([p[1] for p in egc_test_parts_all], axis=0)

oof_r_egc  = np.mean(egc_oof_ridge_all, axis=0)
test_r_egc = np.mean(egc_test_ridge_all, axis=0)

print(f'\nSeed-averaged OOF R²  LGBM={r2_score(y_egc, oof_l_egc):.4f}  '
      f'XGB={r2_score(y_egc, oof_x_egc):.4f}  ChemBERTa-Ridge={r2_score(y_egc, oof_r_egc):.4f}')


# -----------------------------------------------------------------------------
# 6. ENSEMBLE WEIGHT OPTIMIZATION
# -----------------------------------------------------------------------------
from scipy.optimize import minimize

def find_best_weights(parts, y_true):
    stack = np.column_stack(parts)
    def neg_r2(w):
        return -r2_score(y_true, stack @ w)
    n = len(parts)
    res = minimize(
        neg_r2, x0=[1/n]*n, method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
    )
    return res.x

w_tg  = find_best_weights((oof_l_tg, oof_x_tg, oof_g_tg, oof_r_tg),   y_tg)
w_egc = find_best_weights((oof_l_egc, oof_x_egc, oof_g_egc, oof_r_egc), y_egc)

print(f'Optimal Tg  weights — LGBM: {w_tg[0]:.3f}  XGB: {w_tg[1]:.3f}  GNN: {w_tg[2]:.3f}  ChemB: {w_tg[3]:.3f}')
print(f'Optimal Egc weights — LGBM: {w_egc[0]:.3f}  XGB: {w_egc[1]:.3f}  GNN: {w_egc[2]:.3f}  ChemB: {w_egc[3]:.3f}')

oof_tg_opt  = w_tg[0]*oof_l_tg   + w_tg[1]*oof_x_tg   + w_tg[2]*oof_g_tg   + w_tg[3]*oof_r_tg
oof_egc_opt = w_egc[0]*oof_l_egc  + w_egc[1]*oof_x_egc  + w_egc[2]*oof_g_egc  + w_egc[3]*oof_r_egc

r2_tg  = r2_score(y_tg,  oof_tg_opt)
r2_egc = r2_score(y_egc, oof_egc_opt)

print(f'\nOOF R² Tg  : {r2_tg:.4f}   (v8: 0.9082+)')
print(f'OOF R² Egc : {r2_egc:.4f}   (v8: 0.9151+)')
print(f'Mean OOF R²: {(r2_tg + r2_egc)/2:.4f}')

pred_tg  = w_tg[0]*test_l_tg   + w_tg[1]*test_x_tg   + w_tg[2]*test_g_tg   + w_tg[3]*test_r_tg
pred_egc = w_egc[0]*test_l_egc  + w_egc[1]*test_x_egc  + w_egc[2]*test_g_egc  + w_egc[3]*test_r_egc

print('\nTest predictions updated with optimised 4-way weights.')

# -----------------------------------------------------------------------------
# 7. SUBMISSION CREATION
# -----------------------------------------------------------------------------
sub_tg          = test_tg[['id']].copy()
sub_tg['target'] = pred_tg

sub_egc          = test_egc[['id']].copy()
sub_egc['target'] = pred_egc

submission = (
    pd.concat([sub_tg, sub_egc], axis=0)
    .sort_values('id')
    .reset_index(drop=True)
)

assert submission.shape[0] == len(test), 'Row count mismatch!'
assert submission['target'].isna().sum() == 0, 'NaN in predictions!'

os.makedirs('outputs', exist_ok=True)
submission.to_csv('outputs/submission.csv', index=False)
print('Submission shape:', submission.shape)
print(submission.head(10))
print('\noutputs/submission.csv saved.')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_tg, oof_tg_opt, alpha=0.25, s=8)
lo, hi = min(y_tg.min(), oof_tg_opt.min()), max(y_tg.max(), oof_tg_opt.max())
axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1.5)
axes[0].set_xlabel('True Tg (°C)', fontsize=12)
axes[0].set_ylabel('Pred Tg (°C)', fontsize=12)
axes[0].set_title(f'Tg OOF  R² = {r2_tg:.4f}', fontsize=13)
axes[0].grid(alpha=0.3)

axes[1].scatter(y_egc, oof_egc_opt, alpha=0.25, s=8, color='darkorange')
lo, hi = min(y_egc.min(), oof_egc_opt.min()), max(y_egc.max(), oof_egc_opt.max())
axes[1].plot([lo, hi], [lo, hi], 'r--', lw=1.5)
axes[1].set_xlabel('True Egc (eV)', fontsize=12)
axes[1].set_ylabel('Pred Egc (eV)', fontsize=12)
axes[1].set_title(f'Egc OOF  R² = {r2_egc:.4f}', fontsize=13)
axes[1].grid(alpha=0.3)

plt.suptitle(
    f'v9 (with ChemBERTa) OOF Predicted vs Actual  |  Mean R² = {(r2_tg+r2_egc)/2:.4f}',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/oof_scatter_v9.png', dpi=120, bbox_inches='tight')
plt.show()
print('Plot saved as outputs/oof_scatter_v9.png.')
