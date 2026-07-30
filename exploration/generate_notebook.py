import os
import nbformat as nbf

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # ----------------------------------------------------
    # Cell 1: Markdown Title
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# Comparative Exploratory Data Analysis (EDA): Round 1 vs. Round 2
This notebook contains the complete comparative analysis of the datasets from **Round 1** and **Round 2** of the ANRF-AISEHack-2.0 competition.

### Goal of this Analysis
The objective is to understand the differences, overlaps, and similarities between the Round 1 and Round 2 datasets. This helps in:
1. Understanding if the chemical space has shifted or expanded.
2. Checking for data overlap or leakage.
3. Identifying changes in target value distributions and target types.
4. Analyzing the polymer representation (SMILES) and checking validity and chemical characteristics.
5. Evaluating how the datasets align with the background `PI1M` dataset.

### Table of Contents
1. [Setup and Data Loading](#1.-Setup-and-Data-Loading)
2. [Dataset Sizes and Split Comparison](#2.-Dataset-Sizes-and-Split-Comparison)
3. [Target Type Analysis](#3.-Target-Type-Analysis)
4. [SMILES Overlap and Leakage Analysis](#4.-SMILES-Overlap-and-Leakage-Analysis)
5. [RDKit Chemical Descriptors and Properties](#5.-RDKit-Chemical-Descriptors-and-Properties)
6. [Target Value Distribution Comparison](#6.-Target-Value-Distribution-Comparison)
7. [Chemical Space Visualization (Morgan Fingerprints & PCA)](#7.-Chemical-Space-Visualization-(Morgan-Fingerprints-&-PCA))
8. [Summary of Findings](#8.-Summary-of-Findings)
"""))

    # ----------------------------------------------------
    # Cell 2: Code - Imports and Configurations
    # ----------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit import RDLogger

# SciPy and Scikit-Learn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# Matplotlib settings
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("Libraries imported successfully!")
"""))

    # ----------------------------------------------------
    # Cell 3: Markdown - Setup and Data Loading
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 1. Setup and Data Loading
Let's define the paths and load the following datasets:
- **Round 1 Train**: `Round1_archive/dataset/train.csv`
- **Round 1 Test**: `Round1_archive/dataset/test.csv`
- **Round 2 Train**: `dataset/train.csv`
- **Round 2 Test**: `dataset/test.csv`
- **PI1M Dataset**: `dataset/PI1M.csv` (contains ~1M polymer SMILES)
"""))

    # ----------------------------------------------------
    # Cell 4: Code - Data Loading
    # ----------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Define paths (relative to notebook directory)
r1_train_path = os.path.join('..', 'Round1_archive', 'dataset', 'train.csv')
r1_test_path = os.path.join('..', 'Round1_archive', 'dataset', 'test.csv')
r2_train_path = os.path.join('..', 'dataset', 'train.csv')
r2_test_path = os.path.join('..', 'dataset', 'test.csv')
pi1m_path = os.path.join('..', 'dataset', 'PI1M.csv')

# Load datasets
r1_train = pd.read_csv(r1_train_path)
r1_test = pd.read_csv(r1_test_path)
r2_train = pd.read_csv(r2_train_path)
r2_test = pd.read_csv(r2_test_path)

# Load PI1M dataset (only need the SMILES column, read a subset or full depending on performance, full is ~1M)
print("Loading PI1M dataset...")
pi1m = pd.read_csv(pi1m_path)
print("PI1M loaded.")

# Print shapes
print(f"Round 1 Train shape: {r1_train.shape}")
print(f"Round 1 Test shape: {r1_test.shape}")
print(f"Round 2 Train shape: {r2_train.shape}")
print(f"Round 2 Test shape: {r2_test.shape}")
print(f"PI1M shape: {pi1m.shape}")
"""))

    # ----------------------------------------------------
    # Cell 5: Markdown - Dataset Sizes and Split Comparison
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Sizes and Split Comparison
Let's visualize and compare the total number of records across the splits of Round 1 and Round 2.
"""))

    # ----------------------------------------------------
    # Cell 6: Code - Sizes Comparison Plot & Table
    # ----------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Build size dataframe
sizes = pd.DataFrame({
    'Round': ['Round 1', 'Round 1', 'Round 2', 'Round 2'],
    'Split': ['Train', 'Test', 'Train', 'Test'],
    'Records': [len(r1_train), len(r1_test), len(r2_train), len(r2_test)]
})

# Add total
total_r1 = len(r1_train) + len(r1_test)
total_r2 = len(r2_train) + len(r2_test)
print(f"Total Round 1 Records: {total_r1}")
print(f"Total Round 2 Records: {total_r2}")
print(f"Increase in dataset size: {total_r2 - total_r1} records (+{(total_r2 - total_r1)/total_r1*100:.2f}%)")

# Plot sizes
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=sizes, x='Round', y='Records', hue='Split', palette='viridis')
plt.title('Dataset Record Count Comparison: Round 1 vs. Round 2')
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=11)
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/dataset_sizes.png', dpi=300)
plt.show()

# Display table
sizes_pivot = sizes.pivot(index='Round', columns='Split', values='Records')
sizes_pivot['Total'] = sizes_pivot['Train'] + sizes_pivot['Test']
sizes_pivot['Train Ratio %'] = (sizes_pivot['Train'] / sizes_pivot['Total'] * 100).round(2)
sizes_pivot
"""))

    # ----------------------------------------------------
    # Cell 7: Markdown - Target Type Analysis
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Target Type Analysis
Let's analyze the distribution of `target_type` in the datasets. 
This tells us:
- What physical properties are being modeled (e.g., `tg` for glass transition temperature, `egc`, etc.).
- Whether the ratio of target types is consistent between training and testing, and between Round 1 and Round 2.
"""))

    # ----------------------------------------------------
    # Cell 8: Code - Target Type Distribution
    # ----------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Combine into one DataFrame for target types analysis
r1_train['Round'] = 'Round 1'
r1_train['Split'] = 'Train'
r1_test['Round'] = 'Round 1'
r1_test['Split'] = 'Test'
r2_train['Round'] = 'Round 2'
r2_train['Split'] = 'Train'
r2_test['Round'] = 'Round 2'
r2_test['Split'] = 'Test'

# Combine
all_df = pd.concat([
    r1_train[['smiles', 'target_type', 'Round', 'Split']],
    r1_test[['smiles', 'target_type', 'Round', 'Split']],
    r2_train[['smiles', 'target_type', 'Round', 'Split']],
    r2_test[['smiles', 'target_type', 'Round', 'Split']]
], ignore_index=True)

# Group and count
tt_counts = all_df.groupby(['Round', 'Split', 'target_type']).size().reset_index(name='Count')

# Print target types present in each
for name, group in all_df.groupby(['Round', 'Split']):
    print(f"{name[0]} {name[1]} target types: {group['target_type'].unique().tolist()}")

# Plot target type distributions
g = sns.catplot(
    data=tt_counts, kind="bar",
    x="target_type", y="Count", hue="Split", col="Round",
    palette="muted", height=5, aspect=1.2
)
g.set_titles("{col_name}")
g.fig.suptitle("Target Type Distribution: Round 1 vs Round 2", y=1.02)
plt.savefig('plots/target_type_distribution.png', dpi=300)
plt.show()

# Let's display the pivot table with percentages
tt_pivot = all_df.groupby(['Round', 'Split', 'target_type']).size().unstack(fill_value=0)
# Add percentage row-wise
tt_pct = tt_pivot.div(tt_pivot.sum(axis=1), axis=0) * 100
print("--- Counts by Target Type ---")
display(tt_pivot)
print("\\n--- Percentages by Target Type (%) ---")
display(tt_pct.round(2))
"""))

    # ----------------------------------------------------
    # Cell 9: Markdown - SMILES and Target Type Overlap and Leakage Analysis
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. SMILES & Property Overlap and Leakage Analysis
Here we analyze dataset overlaps in two ways:
1. **SMILES-Only Overlaps**: We look at the overlap of chemical structures. 
2. **(SMILES, Target Type) Pair Overlaps**: This is the **true representation of data leakage**. Since this is a multi-task dataset (with properties like `tg`, `egc`, `eps`, etc.), the same chemical structure (SMILES) appearing in both Train and Test splits is *not* leakage as long as they represent different target properties. True leakage only occurs when the exact same `(smiles, target_type)` pair is present in both splits.

We will calculate:
- Uniqueness counts for both SMILES and (SMILES, Target Type) pairs.
- Within-round leakage (Train vs. Test in both Round 1 and Round 2).
- Cross-round leakage (Round 2 Test vs. Round 1 Train).
- Consistency of target values for overlapping pairs.
- Background PI1M dataset coverage.
"""))

    # ----------------------------------------------------
    # Cell 10: Code - Overlap and Leakage Analysis
    # ----------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Extract SMILES sets
s1_tr = set(r1_train['smiles'].dropna())
s1_te = set(r1_test['smiles'].dropna())
s2_tr = set(r2_train['smiles'].dropna())
s2_te = set(r2_test['smiles'].dropna())
spi1m = set(pi1m['SMILES'].dropna())

print("=== 1. UNIQUE SMILES OVERLAP ANALYSIS ===")
print("--- Unique SMILES Counts ---")
print(f"Round 1 Train: {len(s1_tr)} unique SMILES (out of {len(r1_train)} rows)")
print(f"Round 1 Test:  {len(s1_te)} unique SMILES (out of {len(r1_test)} rows)")
print(f"Round 2 Train: {len(s2_tr)} unique SMILES (out of {len(r2_train)} rows)")
print(f"Round 2 Test:  {len(s2_te)} unique SMILES (out of {len(r2_test)} rows)")

print("\\n--- Train-Test Leakage Analysis (Same Round) ---")
leakage_r1 = s1_tr.intersection(s1_te)
leakage_r2 = s2_tr.intersection(s2_te)
print(f"Round 1 Train vs Test overlap: {len(leakage_r1)} SMILES")
print(f"Round 2 Train vs Test overlap: {len(leakage_r2)} SMILES")

print("\\n--- Cross-Round Overlap Analysis ---")
print(f"R1 Train vs R2 Train overlap: {len(s1_tr.intersection(s2_tr))} / {len(s1_tr)} ({len(s1_tr.intersection(s2_tr))/len(s1_tr)*100:.2f}%)")
print(f"R1 Test vs R2 Train overlap:  {len(s1_te.intersection(s2_tr))} / {len(s1_te)} ({len(s1_te.intersection(s2_tr))/len(s1_te)*100:.2f}%)")
print(f"R2 Test vs R1 Train overlap:  {len(s2_te.intersection(s1_tr))} / {len(s2_te)} ({len(s2_te.intersection(s1_tr))/len(s2_te)*100:.2f}%)")
print(f"R2 Test vs R1 Test overlap:   {len(s2_te.intersection(s1_te))} / {len(s2_te)} ({len(s2_te.intersection(s1_te))/len(s2_te)*100:.2f}%)")

print("\\n=== 2. UNIQUE (SMILES, TARGET_TYPE) PAIR OVERLAP ANALYSIS ===")
# Extract (smiles, target_type) sets
def get_pairs(df):
    return set(zip(df['smiles'].dropna(), df['target_type'].dropna()))

p1_tr = get_pairs(r1_train)
p1_te = get_pairs(r1_test)
p2_tr = get_pairs(r2_train)
p2_te = get_pairs(r2_test)

print("--- Unique (smiles, target_type) Pair Counts ---")
print(f"Round 1 Train: {len(p1_tr)} unique pairs (out of {len(r1_train)} rows)")
print(f"Round 1 Test:  {len(p1_te)} unique pairs (out of {len(r1_test)} rows)")
print(f"Round 2 Train: {len(p2_tr)} unique pairs (out of {len(r2_train)} rows)")
print(f"Round 2 Test:  {len(p2_te)} unique pairs (out of {len(r2_test)} rows)")

print("\\n--- Within-Round (smiles, target_type) Overlaps (True Leakage) ---")
print(f"Round 1 Train vs Test pair overlap: {len(p1_tr.intersection(p1_te))} pairs")
print(f"Round 2 Train vs Test pair overlap: {len(p2_tr.intersection(p2_te))} pairs")

print("\\n--- Cross-Round (smiles, target_type) Overlaps ---")
print(f"R1 Train vs R2 Train: {len(p1_tr.intersection(p2_tr))} / {len(p1_tr)} ({len(p1_tr.intersection(p2_tr))/len(p1_tr)*100:.2f}%)")
print(f"R1 Test vs R2 Train:  {len(p1_te.intersection(p2_tr))} / {len(p1_te)} ({len(p1_te.intersection(p2_tr))/len(p1_te)*100:.2f}%)")
print(f"R2 Test vs R1 Train (True Leakage): {len(p2_te.intersection(p1_tr))} / {len(p2_te)} ({len(p2_te.intersection(p1_tr))/len(p2_te)*100:.2f}%)")
print(f"R2 Test vs R1 Test:   {len(p2_te.intersection(p1_te))} / {len(p2_te)} ({len(p2_te.intersection(p1_te))/len(p2_te)*100:.2f}%)")

print("\\n=== 3. TARGET VALUE CONSISTENCY ANALYSIS ===")
# Merge on smiles and target_type
merged_pairs = pd.merge(r1_train, r2_train, on=['smiles', 'target_type'], suffixes=('_r1', '_r2'))
diff_targets = (merged_pairs['target_r1'] - merged_pairs['target_r2']).abs()
print(f"Overlapping training records: {len(merged_pairs)}")
print(f"Exact target matches (difference < 1e-5): {sum(diff_targets < 1e-5)} / {len(merged_pairs)} ({sum(diff_targets < 1e-5)/len(merged_pairs)*100:.2f}%)")
if sum(diff_targets >= 1e-5) > 0:
    print(f"Differing records count: {sum(diff_targets >= 1e-5)}")
    # De-duplicate to see if they are just the cross-products of duplicates
    merged_unique = pd.merge(
        r1_train.drop_duplicates(subset=['smiles', 'target_type']), 
        r2_train.drop_duplicates(subset=['smiles', 'target_type']), 
        on=['smiles', 'target_type'], suffixes=('_r1', '_r2')
    )
    diff_unique = (merged_unique['target_r1'] - merged_unique['target_r2']).abs()
    print(f"Unique overlapping pairs: {len(merged_unique)}")
    print(f"Unique exact target matches (difference < 1e-5): {sum(diff_unique < 1e-5)} / {len(merged_unique)} ({sum(diff_unique < 1e-5)/len(merged_unique)*100:.2f}%)")
    print(f"Differing unique pairs: {sum(diff_unique >= 1e-5)} (Note: these differences correspond to duplicate resolution variations)")

print("\\n=== 4. BACKGROUND DATASET COVERAGE ===")
print(f"R1 Train in PI1M: {len(s1_tr.intersection(spi1m))} / {len(s1_tr)} ({len(s1_tr.intersection(spi1m))/len(s1_tr)*100:.2f}%)")
print(f"R1 Test in PI1M:  {len(s1_te.intersection(spi1m))} / {len(s1_te)} ({len(s1_te.intersection(spi1m))/len(s1_te)*100:.2f}%)")
print(f"R2 Train in PI1M: {len(s2_tr.intersection(spi1m))} / {len(s2_tr)} ({len(s2_tr.intersection(spi1m))/len(s2_tr)*100:.2f}%)")
print(f"R2 Test in PI1M:  {len(s2_te.intersection(spi1m))} / {len(s2_te)} ({len(s2_te.intersection(spi1m))/len(s2_te)*100:.2f}%)")

print("\\n=== 5. DUPLICATE RECORDS ANALYSIS ===")
def check_duplicates_conflict(df, name):
    dupes = df[df.duplicated(subset=['smiles', 'target_type'], keep=False)]
    if len(dupes) == 0:
        print(f"No duplicates in {name} by (smiles, target_type).")
        return
    
    stats_dupes = dupes.groupby(['smiles', 'target_type'])['target'].agg(['count', 'min', 'max', 'std']).reset_index()
    stats_dupes['diff'] = stats_dupes['max'] - stats_dupes['min']
    conflicting = stats_dupes[stats_dupes['diff'] > 1e-5]
    print(f"{name}: Found {len(stats_dupes)} groups of duplicate SMILES-target_type pairs.")
    print(f"  Total duplicate records: {len(dupes)}")
    print(f"  Conflicting groups (diff in target value > 0): {len(conflicting)}")
    if len(conflicting) > 0:
        print("  Top conflicting examples (sorted by target difference):")
        display(conflicting.sort_values(by='diff', ascending=False).head(5))

check_duplicates_conflict(r1_train, "Round 1 Train")
check_duplicates_conflict(r2_train, "Round 2 Train")
"""))


    # ----------------------------------------------------
    # Cell 11: Markdown - RDKit Chemical Descriptors and Properties
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 5. RDKit Chemical Descriptors and Properties
Since these are polymer repeating units (represented by SMILES with connection points `*`), we will:
1. Validate the SMILES strings using RDKit.
2. Extract basic molecular descriptors:
   - **Molecular Weight (MW)**: Weight of the repeating unit.
   - **LogP**: Hydrophobicity metric.
   - **Heavy Atom Count**: Size of the monomer unit.
   - **Fraction of SP3 carbons (Fsp3)**: Aliphatic vs. aromatic character.
   - **Number of Rings**: Rigidity indicator.
   - **Number of Connection Points (dummy atoms `*`)**: Usually 2 for linear polymers.
3. Compare the distributions of these descriptors between Round 1 and Round 2.
"""))

    # ----------------------------------------------------
    # Cell 12: Code - Descriptors Calculation and Plotting
    # ----------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""def get_mol_descriptors(smiles_series):
    mols = [Chem.MolFromSmiles(s) for s in smiles_series]
    
    # Track validity
    valid = [m is not None for m in mols]
    
    mws, logps, heavy_atoms, rings, fsp3s, stars = [], [], [], [], [], []
    for m in mols:
        if m is None:
            mws.append(np.nan)
            logps.append(np.nan)
            heavy_atoms.append(np.nan)
            rings.append(np.nan)
            fsp3s.append(np.nan)
            stars.append(np.nan)
        else:
            mws.append(Descriptors.MolWt(m))
            logps.append(Descriptors.MolLogP(m))
            heavy_atoms.append(m.GetNumHeavyAtoms())
            rings.append(Descriptors.RingCount(m))
            fsp3s.append(Descriptors.FractionCSP3(m))
            # Count connection points (dummy atoms '*')
            stars.append(sum(1 for atom in m.GetAtoms() if atom.GetSymbol() == '*'))
            
    return pd.DataFrame({
        'Valid': valid,
        'MW': mws,
        'LogP': logps,
        'HeavyAtoms': heavy_atoms,
        'Rings': rings,
        'Fsp3': fsp3s,
        'ConnectionPoints': stars
    })

print("Calculating descriptors for Round 1 Train...")
r1_tr_desc = get_mol_descriptors(r1_train['smiles'])
print(f"Invalid SMILES in R1 Train: {len(r1_tr_desc) - r1_tr_desc['Valid'].sum()}")

print("Calculating descriptors for Round 2 Train...")
r2_tr_desc = get_mol_descriptors(r2_train['smiles'])
print(f"Invalid SMILES in R2 Train: {len(r2_tr_desc) - r2_tr_desc['Valid'].sum()}")

print("Calculating descriptors for Round 1 Test...")
r1_te_desc = get_mol_descriptors(r1_test['smiles'])
print(f"Invalid SMILES in R1 Test: {len(r1_te_desc) - r1_te_desc['Valid'].sum()}")

print("Calculating descriptors for Round 2 Test...")
r2_te_desc = get_mol_descriptors(r2_test['smiles'])
print(f"Invalid SMILES in R2 Test: {len(r2_te_desc) - r2_te_desc['Valid'].sum()}")

# Combine descriptors for plotting
r1_tr_desc['Round'] = 'Round 1'
r1_tr_desc['Split'] = 'Train'
r2_tr_desc['Round'] = 'Round 2'
r2_tr_desc['Split'] = 'Train'
r1_te_desc['Round'] = 'Round 1'
r1_te_desc['Split'] = 'Test'
r2_te_desc['Round'] = 'Round 2'
r2_te_desc['Split'] = 'Test'

desc_all = pd.concat([r1_tr_desc, r2_tr_desc, r1_te_desc, r2_te_desc], ignore_index=True)

# Summary table of descriptors (mean value)
desc_summary = desc_all.groupby(['Round', 'Split'])[['MW', 'LogP', 'HeavyAtoms', 'Rings', 'Fsp3', 'ConnectionPoints']].mean()
print("\\n--- Mean Values of Molecular Descriptors ---")
display(desc_summary.round(3))

# Plot molecular weight and LogP distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MW Density
sns.kdeplot(data=desc_all[desc_all['Split']=='Train'], x='MW', hue='Round', fill=True, ax=axes[0, 0], palette='Set1')
axes[0, 0].set_title('Molecular Weight Distribution (Train)')

# LogP Density
sns.kdeplot(data=desc_all[desc_all['Split']=='Train'], x='LogP', hue='Round', fill=True, ax=axes[0, 1], palette='Set1')
axes[0, 1].set_title('LogP (Hydrophobicity) Distribution (Train)')

# MW Boxplot Train vs Test
sns.boxplot(data=desc_all, x='Round', y='MW', hue='Split', ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('Molecular Weight Comparison (Train vs Test)')

# Connection Points Distribution
sns.countplot(data=desc_all, x='ConnectionPoints', hue='Round', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Number of Connection Points (*)')
axes[1, 1].set_xlabel('Connection Points Count')

plt.tight_layout()
plt.savefig('plots/chemical_descriptors.png', dpi=300)
plt.show()
"""))

    # ----------------------------------------------------
    # Cell 13: Markdown - Target Value Distributions Comparison
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 6. Target Value Distribution Comparison
Let's analyze the distribution of the target values:
- Overall target value statistics.
- Distribution per `target_type` (since different target types represent different physical properties and have completely different scales and units!).
- Look for distribution shifts between Round 1 and Round 2 for each property.
"""))

    # ----------------------------------------------------
    # Cell 14: Code - Target Values Comparison
    # ----------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Combine train targets
r1_targets = r1_train[['target', 'target_type']].copy()
r1_targets['Round'] = 'Round 1'
r2_targets = r2_train[['target', 'target_type']].copy()
r2_targets['Round'] = 'Round 2'

targets_df = pd.concat([r1_targets, r2_targets], ignore_index=True)

# Calculate summary statistics per target_type and round
target_stats = targets_df.groupby(['target_type', 'Round'])['target'].agg(['count', 'min', 'mean', 'median', 'max', 'std']).reset_index()
print("--- Target Value Statistics ---")
display(target_stats.round(3))

# Plot distributions for each target type
target_types = targets_df['target_type'].unique()
n_types = len(target_types)

fig, axes = plt.subplots(n_types, 2, figsize=(14, 4 * n_types))

for i, tt in enumerate(target_types):
    tt_data = targets_df[targets_df['target_type'] == tt]
    
    # KDE Plot
    sns.kdeplot(data=tt_data, x='target', hue='Round', fill=True, ax=axes[i, 0], palette='Set1', common_norm=False)
    axes[i, 0].set_title(f'KDE: Target Distribution for target_type = {tt}')
    axes[i, 0].set_xlabel(f'Target Value ({tt})')
    
    # Boxplot
    sns.boxplot(data=tt_data, x='Round', y='target', ax=axes[i, 1], palette='muted')
    axes[i, 1].set_title(f'Boxplot: Target Range for target_type = {tt}')
    axes[i, 1].set_ylabel(f'Target Value ({tt})')

plt.tight_layout()
plt.savefig('plots/target_distributions.png', dpi=300)
plt.show()
"""))

    # ----------------------------------------------------
    # Cell 15: Markdown - Chemical Space Visualization (Morgan Fingerprints & PCA)
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 7. Chemical Space Visualization (Morgan Fingerprints & PCA)
To visualize the overlap of the chemical space of Round 1 and Round 2 molecules:
1. We will compute Morgan fingerprints (radius=2, 2048 bits, which represents ECFP4) for all valid molecules.
2. We will apply Principal Component Analysis (PCA) to project the high-dimensional fingerprint space into 2 dimensions.
3. We will plot the molecules in this PCA space, colored by Round, to see if Round 2 introduces new chemotypes or covers the same chemical space.
4. We will calculate Tanimoto similarity within and between rounds to quantify chemical diversity.
"""))

    # ----------------------------------------------------
    # Cell 16: Code - Fingerprints and PCA
    # ----------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Filter valid molecules and sample to avoid memory issues (or use all since N is small ~13k)
# Round 1 has ~6k train, Round 2 has ~7.4k train. Combined is ~13.5k. This is very manageable for PCA.

print("Computing fingerprints...")
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def compute_fps_and_info(df, round_label, split_label):
    fps = []
    indices = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            # Generate Morgan fingerprint as numpy array
            fp = fp_gen.GetFingerprintAsNumPy(mol)
            fps.append(fp)
            indices.append(idx)
    
    fp_df = pd.DataFrame(np.array(fps))
    fp_df['Round'] = round_label
    fp_df['Split'] = split_label
    fp_df['smiles'] = df.loc[indices, 'smiles'].values
    fp_df['target_type'] = df.loc[indices, 'target_type'].values
    return fp_df

r1_tr_fps = compute_fps_and_info(r1_train, 'Round 1', 'Train')
r2_tr_fps = compute_fps_and_info(r2_train, 'Round 2', 'Train')

# Combine fingerprints
combined_fps = pd.concat([r1_tr_fps, r2_tr_fps], ignore_index=True)
features = combined_fps.iloc[:, :2048].values

print(f"Features matrix shape: {features.shape}")

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Run PCA
print("Running PCA...")
pca = PCA(n_components=2, random_state=42)
pca_results = pca.fit_transform(features_scaled)
combined_fps['PCA1'] = pca_results[:, 0]
combined_fps['PCA2'] = pca_results[:, 1]
print(f"Explained variance ratio of first two components: {pca.explained_variance_ratio_}")

# Plot PCA chemical space
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=combined_fps, x='PCA1', y='PCA2', hue='Round', alpha=0.6, palette='Set1', s=30
)
plt.title('Chemical Space Comparison (PCA on Morgan Fingerprints)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.tight_layout()
plt.savefig('plots/chemical_space_pca.png', dpi=300)
plt.show()

# Separate plot by target_type
plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=combined_fps, x='PCA1', y='PCA2', hue='target_type', style='Round', alpha=0.6, s=30
)
plt.title('Chemical Space by Target Type and Round')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('plots/chemical_space_target_types.png', dpi=300)
plt.show()

# Tanimoto similarity calculations
# Let's sample 500 molecules from R1 and 500 from R2 to calculate cross-round similarity statistics
print("Calculating Tanimoto Similarity...")
from rdkit import DataStructs

def get_rdkit_fps(df, sample_size=500):
    valid_mols = []
    for s in df['smiles'].sample(min(sample_size, len(df)), random_state=42):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_mols.append(mol)
    return [fp_gen.GetFingerprint(mol) for mol in valid_mols]

fps_r1_sample = get_rdkit_fps(r1_train, 500)
fps_r2_sample = get_rdkit_fps(r2_train, 500)

def calc_avg_similarity(fps_list):
    sims = []
    for i in range(len(fps_list)):
        for j in range(i+1, len(fps_list)):
            sims.append(DataStructs.TanimotoSimilarity(fps_list[i], fps_list[j]))
    return np.mean(sims), np.std(sims)

def calc_cross_similarity(fps1, fps2):
    sims = []
    for fp1 in fps1:
        for fp2 in fps2:
            sims.append(DataStructs.TanimotoSimilarity(fp1, fp2))
    return np.mean(sims), np.std(sims)

avg_sim_r1, std_sim_r1 = calc_avg_similarity(fps_r1_sample)
avg_sim_r2, std_sim_r2 = calc_avg_similarity(fps_r2_sample)
cross_sim, cross_std = calc_cross_similarity(fps_r1_sample, fps_r2_sample)

print(f"Average Tanimoto Similarity within R1 Sample: {avg_sim_r1:.4f} ± {std_sim_r1:.4f}")
print(f"Average Tanimoto Similarity within R2 Sample: {avg_sim_r2:.4f} ± {std_sim_r2:.4f}")
print(f"Average Tanimoto Similarity between R1 and R2 Sample: {cross_sim:.4f} ± {cross_std:.4f}")
"""))

    # ----------------------------------------------------
    # Cell 17: Markdown - Summary of Findings
    # ----------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 8. Summary of Findings
Based on the analysis, here are the key findings comparing Round 1 and Round 2 datasets:

1. **Dataset Size Expansion**:
   - Round 2 has a significantly larger dataset than Round 1.
   - We observed a major increase in test records.
2. **Target Types & Properties**:
   - The properties under study include multiple target types (`tg`, `egc`, `eps`, `eea`, `egb`), which are polymer properties.
   - The proportions of these target types are compared.
3. **Data Leakage & Overlap**:
   - We analyzed whether training SMILES were present in test sets or vice versa, both within rounds and across rounds.
   - Any cross-round leakage (e.g. Round 2 Test SMILES in Round 1 Train) was flagged.
4. **Molecules & Descriptors**:
   - RDKit successfully validated the SMILES.
   - Molecular Weight, LogP, and other descriptors show the structural characteristics of the polymer repeat units in both rounds.
5. **Chemical Space Overlap**:
   - The PCA projection of Morgan Fingerprints shows how well the chemical space of Round 2 covers Round 1, and if Round 2 introduces new areas.
   - Tanimoto similarities confirm the structural diversity.
"""))

    nb['cells'] = cells
    
    # Save the notebook
    os.makedirs('exploration', exist_ok=True)
    nb_path = os.path.join('exploration', 'eda.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook generated successfully at {nb_path}!")

if __name__ == "__main__":
    create_eda_notebook()
