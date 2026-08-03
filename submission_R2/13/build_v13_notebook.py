import json

# ============================================================
# v13 Notebook Builder
# Combines NB04 (GPU MultiTask NN as Stage A) with NB03 (Stage B sibling stacking)
# into a single self-contained Kaggle notebook.
# ============================================================

def extract_code_cells(nb):
    """Extract (source, metadata) from all code cells."""
    cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            cells.append(source)
    return cells

def make_cell(source, cell_type='code'):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if cell_type == 'markdown' else source,
        "outputs": [],
        "execution_count": None,
    }

def make_md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }

# ----- Load both notebooks -----
with open(r'C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\Shivesh\anrf-app04.ipynb', 'r', encoding='utf-8') as f:
    nb04 = json.load(f)

with open(r'C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\Shivesh\anrf-app3.ipynb', 'r', encoding='utf-8') as f:
    nb03 = json.load(f)

# ----- Extract code from NB04 -----
nb04_cells = nb04['cells']

# Common library (shared between both) - take from NB04
common_lib_cell = None
data_path_cell = None
psmiles_cell = None
featurize_cell = None
cv_metric_cell = None
artifacts_cell = None
override_cell = None

for cell in nb04_cells:
    if cell['cell_type'] != 'code':
        continue
    src = "".join(cell['source'])
    if 'SHARED LIBRARY' in src and common_lib_cell is None:
        common_lib_cell = src
    if '_iter_csvs' in src and data_path_cell is None:
        data_path_cell = src
    if 'build_oligomer' in src and psmiles_cell is None:
        psmiles_cell = src
    if '_descriptors' in src and featurize_cell is None:
        featurize_cell = src
    if 'make_folds' in src and cv_metric_cell is None:
        cv_metric_cell = src
    if 'save_oof' in src and artifacts_cell is None:
        artifacts_cell = src
    if 'apply_override' in src and override_cell is None:
        override_cell = src

# NB04-specific: torch imports + data loading
nn_setup_cell = None
nn_features_cell = None
nn_target_matrix_cell = None
nn_model_cell = None
nn_train_cell = None
nn_score_emit_cell = None

for cell in nb04_cells:
    if cell['cell_type'] != 'code':
        continue
    src = "".join(cell['source'])
    if 'import torch' in src and nn_setup_cell is None:
        nn_setup_cell = src
    if 'X_all, cols = featurize' in src and nn_features_cell is None:
        nn_features_cell = src
    if 'n_mol = len(uniq)' in src and nn_target_matrix_cell is None:
        nn_target_matrix_cell = src
    if 'class MultiTaskNet' in src and nn_model_cell is None:
        nn_model_cell = src
    if 't0 = time.time()' in src and nn_train_cell is None:
        nn_train_cell = src
    if 'stageA = pd.DataFrame(full_mat' in src and nn_score_emit_cell is None:
        nn_score_emit_cell = src

# NB03-specific: Stage B imports, Stage B sibling tables, features, training, submit
nb03_cells = nb03['cells']

nb03_setup_cell = None
sibling_table_cell = None
make_features_cell = None
stage_b_train_cell = None
assertions_cell = None
score_save_cell = None
submit_cell = None

for cell in nb03_cells:
    if cell['cell_type'] != 'code':
        continue
    src = "".join(cell['source'])
    if 'import scipy.optimize' in src and nb03_setup_cell is None:
        nb03_setup_cell = src
    if 'fold_tables = ' in src and sibling_table_cell is None:
        sibling_table_cell = src
    if 'def make_features' in src and make_features_cell is None:
        make_features_cell = src
    if 'def stage_b_models' in src and stage_b_train_cell is None:
        stage_b_train_cell = src
    if 'fold-safety assertions passed' in src and assertions_cell is None:
        assertions_cell = src
    if 'score_table(train' in src and score_save_cell is None:
        score_save_cell = src
    if 'apply_override(test, train, arch' in src and 'write_submission' in src and submit_cell is None:
        submit_cell = src

# ----- Build the v13 notebook -----
# The key integration point: NB04 already outputs `stageA` as a DataFrame.
# NB03's Stage B reads from the `stageA` DataFrame via make_features().
# So we run all of NB04 first (which creates `stageA`), then jump straight
# into NB03's Stage B (skipping NB03's own Stage A building).

# We need to patch NB03's setup cell to not rebuild Stage A:
# Remove the REUSE_STAGE_A / build_stage_a_compact / find_stage_a block
nb03_setup_patched = """import scipy.optimize as sopt
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

NAME = "v13_nn_sibling"
N_FOLDS = 10
N_AUG = 4          # masked replicas of the training set (robustness augmentation)
FAST = bool(int(os.environ.get("FAST", "0")))

# NOTE: train, test, arch, uniq, pos are already loaded by the NB04 section above.
# Stage A is already built as `stageA` (MultiTask NN output).
# We go straight to Stage B.
"""

# Patch the nn_score_emit_cell to NOT write submission but DO build stageA correctly
# The original cell writes submission.csv standalone, we want only the stageA + oof parts
nn_score_emit_patched = nn_score_emit_cell
# Replace the standalone write_submission call with nothing (we write it at the end)
nn_score_emit_patched = nn_score_emit_patched.replace(
    "write_submission(apply_override(test, train, arch, enable=True))",
    "# submission will be written by Stage B below"
)

cells = []

# Cell 1: Title
cells.append(make_md_cell(
    "# NB-v13: GPU MultiTask NN (Stage A) → Sibling Stacking (Stage B)\n\n"
    "**Single self-contained Kaggle notebook (requires GPU accelerator ON).**\n\n"
    "Architecture:\n"
    "1. **Stage A**: PyTorch MultiTask Neural Network over all 7 properties simultaneously (NB04). "
    "Much stronger than KernelRidge for eps/ei/nc.\n"
    "2. **Stage B**: Sibling-conditioned LightGBM + RidgeCV stacker (NB03). Reads NN Stage A "
    "predictions plus known sibling labels and residual bias features.\n"
    "3. **Override**: ~50% of test rows replaced with exact measured values from archive.\n\n"
    "Expected: OOF R² > 0.893, leaderboard score > 0.899."
))

# Cell 2: Common shared library
cells.append(make_cell(common_lib_cell))

# Cell 3: Data paths
cells.append(make_cell(data_path_cell))

# Cell 4: PSMILES chemistry
cells.append(make_cell(psmiles_cell))

# Cell 5: Featurisation
cells.append(make_cell(featurize_cell))

# Cell 6: CV & metric
cells.append(make_cell(cv_metric_cell))

# Cell 7: Artifacts / submission helpers
cells.append(make_cell(artifacts_cell + "\n\n" + override_cell))

# Cell 8: NB04 setup (torch, data load, pos dict)
cells.append(make_md_cell("## Stage A — GPU MultiTask Neural Network"))
cells.append(make_cell(nn_setup_cell))

# Cell 9: NB04 feature matrix
cells.append(make_cell(nn_features_cell))

# Cell 10: NB04 target matrix
cells.append(make_cell(nn_target_matrix_cell))

# Cell 11: NB04 model definition
cells.append(make_md_cell("### Stage A Model"))
cells.append(make_cell(nn_model_cell))

# Cell 12: NB04 cross-validated training
cells.append(make_md_cell("### Cross-validated training (10-fold × 3 seeds)"))
cells.append(make_cell(nn_train_cell))

# Cell 13: NB04 score + build stageA DataFrame (patched - no standalone submission)
cells.append(make_md_cell("### Build Stage A DataFrame (used by Stage B)"))
cells.append(make_cell(nn_score_emit_patched))

# Cell 14: NB03 Stage B setup (patched - skip rebuilding Stage A)
cells.append(make_md_cell("## Stage B — Sibling-conditioned Stacking"))
cells.append(make_cell(nb03_setup_patched))

# Cell 15: Sibling tables
cells.append(make_md_cell("### Fold-safe sibling label tables"))
cells.append(make_cell(sibling_table_cell))

# Cell 16: Stage B feature construction
cells.append(make_md_cell("### Stage B feature construction"))
cells.append(make_cell(make_features_cell))

# Cell 17: Stage B training + prediction
cells.append(make_md_cell("### Train Stage B (LightGBM + RidgeCV, NNLS blended)"))
cells.append(make_cell(stage_b_train_cell))

# Cell 18: Assertions
cells.append(make_cell(assertions_cell))

# Cell 19: Score and save OOF
cells.append(make_cell(score_save_cell))

# Cell 20: Final submission - explicit call
explicit_submit = 'sub_df = apply_override(test, train, arch, enable=True)\nwrite_submission(sub_df)'
cells.append(make_cell(explicit_submit))

# ----- Assemble notebook -----
nb_v13 = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells
}

out_path = r'C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\13\anrf-v13-nn-sibling.ipynb'
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb_v13, f, indent=1, ensure_ascii=False)

print(f"Written {out_path}")
print(f"Total cells: {len(cells)}")
