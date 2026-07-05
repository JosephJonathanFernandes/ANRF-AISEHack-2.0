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
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup, AutoConfig

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
print('Base imports successful.')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Torch device: {DEVICE}')

# -----------------------------------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------------------------------
train_path = '../../dataset/train.csv'
test_path  = '../../dataset/test.csv'

if not os.path.exists(train_path):
    train_path = '../dataset/train.csv'
    test_path  = '../dataset/test.csv'
    if not os.path.exists(train_path):
        train_path = 'dataset/train.csv'
        test_path  = 'dataset/test.csv'

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

print('Featurizing tabular features...')
X_tg_raw       = featurize(train_tg['smiles'].tolist())
X_egc_raw      = featurize(train_egc['smiles'].tolist())
X_tg_test_raw  = featurize(test_tg['smiles'].tolist())
X_egc_test_raw = featurize(test_egc['smiles'].tolist())

X_tg,  tg_prep  = build_preprocessor(X_tg_raw)
X_egc, egc_prep = build_preprocessor(X_egc_raw)
X_tg_test  = apply_preprocessor(X_tg_test_raw,  tg_prep)
X_egc_test = apply_preprocessor(X_egc_test_raw, egc_prep)

# -----------------------------------------------------------------------------
# 3. TRANSFORMER FINE-TUNING (ENSEMBLE SEEDS + EMBEDDING EXTRACTION)
# -----------------------------------------------------------------------------
class SmilesDataset(Dataset):
    def __init__(self, smiles, tokenizer, targets=None, max_len=128):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_len = max_len
        
    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        encoding = self.tokenizer(
            smi, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if self.targets is not None:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

TRANSFORMER_SEEDS = [42, 7, 123]

def train_transformer_ensemble(df_train, df_test, y_true, target_name, seed, m_name, n_splits=5, epochs=15, batch_size=16, lr=2e-5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(df_train))
    test_preds = np.zeros(len(df_test))
    
    config = AutoConfig.from_pretrained(m_name)
    hidden_dim = config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(m_name)
    
    oof_embs = np.zeros((len(df_train), hidden_dim))
    test_embs = np.zeros((len(df_test), hidden_dim))
    
    test_dataset = SmilesDataset(df_test['smiles'].tolist(), tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        print(f"    Transformer fold {fold+1}/{n_splits} | {target_name} | Seed {seed}", flush=True)
        
        train_smiles = df_train['smiles'].iloc[train_idx].tolist()
        val_smiles   = df_train['smiles'].iloc[val_idx].tolist()
        train_y = y_true[train_idx]
        val_y   = y_true[val_idx]
        
        train_dataset = SmilesDataset(train_smiles, tokenizer, train_y)
        val_dataset   = SmilesDataset(val_smiles, tokenizer, val_y)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = AutoModelForSequenceClassification.from_pretrained(m_name, num_labels=1)
        model.to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        criterion = nn.MSELoss()
        
        best_val_r2 = -float('inf')
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits.view(-1), labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
            model.eval()
            val_preds_fold = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    val_preds_fold.extend(outputs.logits.view(-1).cpu().numpy().tolist())
            
            val_r2 = r2_score(val_y, val_preds_fold)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        model.load_state_dict(best_state)
        model.eval()
        
        val_preds_fold = []
        val_embs_fold = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                val_preds_fold.extend(outputs.logits.view(-1).cpu().numpy().tolist())
                val_embs_fold.extend(outputs.hidden_states[-1][:, 0, :].cpu().numpy())
        oof_preds[val_idx] = val_preds_fold
        oof_embs[val_idx] = np.array(val_embs_fold)
        
        test_preds_fold = []
        test_embs_fold = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                test_preds_fold.extend(outputs.logits.view(-1).cpu().numpy().tolist())
                test_embs_fold.extend(outputs.hidden_states[-1][:, 0, :].cpu().numpy())
        test_preds += np.array(test_preds_fold) / n_splits
        test_embs += np.array(test_embs_fold) / n_splits
        
    print(f"  OOF R² ({target_name} Seed {seed}): {r2_score(y_true, oof_preds):.4f}")
    return oof_preds, test_preds, oof_embs, test_embs

# ----------------- ChemBERTa -----------------
print('\n' + '=' * 60)
print('  ChemBERTa Fine-tuning & Embedding Extraction')
print('=' * 60)
chemberta_model = "DeepChem/ChemBERTa-77M-MTR"

chemberta_oof_tg_all, chemberta_test_tg_all = [], []
chemberta_oof_tg_embs_all, chemberta_test_tg_embs_all = [], []
for seed in TRANSFORMER_SEEDS:
    oof_c, test_c, oof_e, test_e = train_transformer_ensemble(train_tg, test_tg, y_tg, "Tg", seed, chemberta_model)
    chemberta_oof_tg_all.append(oof_c)
    chemberta_test_tg_all.append(test_c)
    chemberta_oof_tg_embs_all.append(oof_e)
    chemberta_test_tg_embs_all.append(test_e)

oof_c_tg  = np.mean(chemberta_oof_tg_all,  axis=0)
test_c_tg = np.mean(chemberta_test_tg_all, axis=0)
oof_ec_tg = np.mean(chemberta_oof_tg_embs_all, axis=0)
test_ec_tg = np.mean(chemberta_test_tg_embs_all, axis=0)

chemberta_oof_egc_all, chemberta_test_egc_all = [], []
chemberta_oof_egc_embs_all, chemberta_test_egc_embs_all = [], []
for seed in TRANSFORMER_SEEDS:
    oof_c, test_c, oof_e, test_e = train_transformer_ensemble(train_egc, test_egc, y_egc, "Egc", seed, chemberta_model)
    chemberta_oof_egc_all.append(oof_c)
    chemberta_test_egc_all.append(test_c)
    chemberta_oof_egc_embs_all.append(oof_e)
    chemberta_test_egc_embs_all.append(test_e)

oof_c_egc  = np.mean(chemberta_oof_egc_all,  axis=0)
test_c_egc = np.mean(chemberta_test_egc_all, axis=0)
oof_ec_egc = np.mean(chemberta_oof_egc_embs_all, axis=0)
test_ec_egc = np.mean(chemberta_test_egc_embs_all, axis=0)


# ----------------- PolyBERT -----------------
print('\n' + '=' * 60)
print('  PolyBERT Fine-tuning & Embedding Extraction')
print('=' * 60)
polybert_model = "xushijie/polyBERT"

polybert_oof_tg_all, polybert_test_tg_all = [], []
polybert_oof_tg_embs_all, polybert_test_tg_embs_all = [], []
for seed in TRANSFORMER_SEEDS:
    oof_p, test_p, oof_e, test_e = train_transformer_ensemble(train_tg, test_tg, y_tg, "Tg", seed, polybert_model)
    polybert_oof_tg_all.append(oof_p)
    polybert_test_tg_all.append(test_p)
    polybert_oof_tg_embs_all.append(oof_e)
    polybert_test_tg_embs_all.append(test_e)

oof_p_tg  = np.mean(polybert_oof_tg_all,  axis=0)
test_p_tg = np.mean(polybert_test_tg_all, axis=0)
oof_ep_tg = np.mean(polybert_oof_tg_embs_all, axis=0)
test_ep_tg = np.mean(polybert_test_tg_embs_all, axis=0)

polybert_oof_egc_all, polybert_test_egc_all = [], []
polybert_oof_egc_embs_all, polybert_test_egc_embs_all = [], []
for seed in TRANSFORMER_SEEDS:
    oof_p, test_p, oof_e, test_e = train_transformer_ensemble(train_egc, test_egc, y_egc, "Egc", seed, polybert_model)
    polybert_oof_egc_all.append(oof_p)
    polybert_test_egc_all.append(test_p)
    polybert_oof_egc_embs_all.append(oof_e)
    polybert_test_egc_embs_all.append(test_e)

oof_p_egc  = np.mean(polybert_oof_egc_all,  axis=0)
test_p_egc = np.mean(polybert_test_egc_all, axis=0)
oof_ep_egc = np.mean(polybert_oof_egc_embs_all, axis=0)
test_ep_egc = np.mean(polybert_test_egc_embs_all, axis=0)


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
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]

    if len(edge_index) == 0:
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

y_tg_scaler  = StandardScaler()
y_egc_scaler = StandardScaler()
y_tg_scaled  = y_tg_scaler.fit_transform(y_tg.reshape(-1, 1)).ravel()
y_egc_scaled = y_egc_scaler.fit_transform(y_egc.reshape(-1, 1)).ravel()

print('\nBuilding graphs...')
graphs_tg = [smiles_to_graph(s, y) for s, y in zip(train_tg['smiles'], y_tg_scaled)]
graphs_tg_test = [smiles_to_graph(s) for s in test_tg['smiles']]
graphs_egc = [smiles_to_graph(s, y) for s, y in zip(train_egc['smiles'], y_egc_scaled)]
graphs_egc_test = [smiles_to_graph(s) for s in test_egc['smiles']]

GNN_SEEDS = [42, 7, 123]

def train_gnn_ensemble(graphs, y_raw, graphs_test, scaler, seed, n_splits=5,
                       hidden=128, layers=3, timesteps=3, epochs=150, patience=20, lr=1e-3):
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

    oof  = scaler.inverse_transform(oof_scaled.reshape(-1, 1)).ravel()
    test = scaler.inverse_transform(test_scaled.reshape(-1, 1)).ravel()
    return oof, test

print('\n' + '=' * 60)
print('  GNN (Large)')
print('=' * 60)

gnn_oof_tg_all, gnn_test_tg_all = [], []
for seed in GNN_SEEDS:
    print(f'  Tg seed={seed}')
    oof_g, test_g = train_gnn_ensemble(graphs_tg, y_tg, graphs_tg_test, y_tg_scaler, seed=seed)
    gnn_oof_tg_all.append(oof_g)
    gnn_test_tg_all.append(test_g)

oof_g_tg  = np.mean(gnn_oof_tg_all,  axis=0)
test_g_tg = np.mean(gnn_test_tg_all, axis=0)

gnn_oof_egc_all, gnn_test_egc_all = [], []
for seed in GNN_SEEDS:
    print(f'  Egc seed={seed}')
    oof_g, test_g = train_gnn_ensemble(graphs_egc, y_egc, graphs_egc_test, y_egc_scaler, seed=seed)
    gnn_oof_egc_all.append(oof_g)
    gnn_test_egc_all.append(test_g)

oof_g_egc  = np.mean(gnn_oof_egc_all,  axis=0)
test_g_egc = np.mean(gnn_test_egc_all, axis=0)


# -----------------------------------------------------------------------------
# 5. TABULAR MODELS (LGBM + XGB) with Transformer Embeddings appended
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

def train_tabular_ensemble(X_train, y_train, X_test, target_type, seed, n_splits=10):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test  = np.asarray(X_test)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_lgbm, test_lgbm = np.zeros(len(X_train)), np.zeros(len(X_test))
    oof_xgb,  test_xgb  = np.zeros(len(X_train)), np.zeros(len(X_test))

    lp = lgbm_params(target_type); lp['random_state'] = seed
    xp = xgb_params(target_type);  xp['random_state'] = seed

    for tr_idx, val_idx in kf.split(X_train):
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

print('\n' + '=' * 60)
print('  Tabular (LGBM+XGB) on Combined Features')
print('=' * 60)

X_tg_comb = np.hstack([X_tg, oof_ec_tg, oof_ep_tg])
X_tg_test_comb = np.hstack([X_tg_test, test_ec_tg, test_ep_tg])

tg_oof_parts_all, tg_test_parts_all = [], []
for seed in SEEDS:
    oof_parts, test_parts = train_tabular_ensemble(X_tg_comb, y_tg, X_tg_test_comb, 'tg', seed=seed)
    tg_oof_parts_all.append(oof_parts)
    tg_test_parts_all.append(test_parts)

oof_l_tg  = np.mean([p[0] for p in tg_oof_parts_all],  axis=0)
oof_x_tg  = np.mean([p[1] for p in tg_oof_parts_all],  axis=0)
test_l_tg = np.mean([p[0] for p in tg_test_parts_all], axis=0)
test_x_tg = np.mean([p[1] for p in tg_test_parts_all], axis=0)

X_egc_comb = np.hstack([X_egc, oof_ec_egc, oof_ep_egc])
X_egc_test_comb = np.hstack([X_egc_test, test_ec_egc, test_ep_egc])

egc_oof_parts_all, egc_test_parts_all = [], []
for seed in SEEDS:
    oof_parts, test_parts = train_tabular_ensemble(X_egc_comb, y_egc, X_egc_test_comb, 'egc', seed=seed)
    egc_oof_parts_all.append(oof_parts)
    egc_test_parts_all.append(test_parts)

oof_l_egc  = np.mean([p[0] for p in egc_oof_parts_all],  axis=0)
oof_x_egc  = np.mean([p[1] for p in egc_oof_parts_all],  axis=0)
test_l_egc = np.mean([p[0] for p in egc_test_parts_all], axis=0)
test_x_egc = np.mean([p[1] for p in egc_test_parts_all], axis=0)

# -----------------------------------------------------------------------------
# 6. ENSEMBLE WEIGHT OPTIMIZATION
# -----------------------------------------------------------------------------
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

w_tg  = find_best_weights((oof_l_tg, oof_x_tg, oof_g_tg, oof_c_tg, oof_p_tg),   y_tg)
w_egc = find_best_weights((oof_l_egc, oof_x_egc, oof_g_egc, oof_c_egc, oof_p_egc), y_egc)

print('\n' + '=' * 60)
print('  v16 ENSEMBLE')
print('=' * 60)
print(f'Optimal Tg  weights — LGBM: {w_tg[0]:.3f}  XGB: {w_tg[1]:.3f}  GNN: {w_tg[2]:.3f}  ChemBERTa: {w_tg[3]:.3f}  PolyBERT: {w_tg[4]:.3f}')
print(f'Optimal Egc weights — LGBM: {w_egc[0]:.3f}  XGB: {w_egc[1]:.3f}  GNN: {w_egc[2]:.3f}  ChemBERTa: {w_egc[3]:.3f}  PolyBERT: {w_egc[4]:.3f}')

oof_tg_opt  = w_tg[0]*oof_l_tg   + w_tg[1]*oof_x_tg   + w_tg[2]*oof_g_tg   + w_tg[3]*oof_c_tg   + w_tg[4]*oof_p_tg
oof_egc_opt = w_egc[0]*oof_l_egc  + w_egc[1]*oof_x_egc  + w_egc[2]*oof_g_egc  + w_egc[3]*oof_c_egc  + w_egc[4]*oof_p_egc

r2_tg  = r2_score(y_tg,  oof_tg_opt)
r2_egc = r2_score(y_egc, oof_egc_opt)

print(f'\nOOF R² Tg  : {r2_tg:.4f}')
print(f'OOF R² Egc : {r2_egc:.4f}')
print(f'Mean OOF R²: {(r2_tg + r2_egc)/2:.4f}')

pred_tg  = w_tg[0]*test_l_tg   + w_tg[1]*test_x_tg   + w_tg[2]*test_g_tg   + w_tg[3]*test_c_tg   + w_tg[4]*test_p_tg
pred_egc = w_egc[0]*test_l_egc  + w_egc[1]*test_x_egc  + w_egc[2]*test_g_egc  + w_egc[3]*test_c_egc  + w_egc[4]*test_p_egc

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

os.makedirs('outputs', exist_ok=True)
submission.to_csv('outputs/submission_v16.csv', index=False)
print('\noutputs/submission_v16.csv saved.')

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

plt.suptitle(f'v16 OOF Predicted vs Actual  |  Mean R² = {(r2_tg+r2_egc)/2:.4f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/oof_scatter_v16.png', dpi=120, bbox_inches='tight')
print('Plot saved as outputs/oof_scatter_v16.png.')
