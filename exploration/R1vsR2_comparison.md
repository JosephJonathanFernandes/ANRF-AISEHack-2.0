# Dataset Comparison Report: Round 1 vs. Round 2

This report presents a detailed comparative analysis between the datasets of **Round 1** (located in `Round1_archive/dataset`) and **Round 2** (located in `dataset`).

The companion notebook containing the complete execution and plots is available at [exploration/eda.ipynb](file:///c:/DEV/Events/aise_hack/ANRF-AISEHack-2.0/exploration/eda.ipynb).

---

## Dataset Folder Structures & File Formats

Before diving into the detailed statistical comparisons, let's examine the files and structural layout of the dataset folders provided in both rounds.

### Directory Layouts
* **Round 1 (`Round1_archive/dataset/`)**:
  * `train.csv` (405 KB): Training dataset containing targets.
  * `test.csv` (264 KB): Test dataset containing SMILES for target prediction.
  * `sample_submission.csv` (102 B): A dummy file showing the submission format.
  * `base_line_model.ipynb` (12 KB): A starter baseline model notebook.
* **Round 2 (`dataset/`)**:
  * `train.csv` (451 KB): Training dataset containing targets.
  * `test.csv` (289 KB): Test dataset containing SMILES for target prediction.
  * `PI1M.csv` (48.6 MB): A large external background dataset containing ~1,000,000 polymer SMILES, which was **not** present in the Round 1 dataset folder.

### File Schema Comparison
The core files in both rounds share the exact same structural schema:
* **`train.csv`**: Same columns in both rounds:
  * `smiles`: SMILES representation of the polymer repeat unit (containing connection points `*`).
  * `target`: Floating-point target value.
  * `target_type`: A categorical string representing the property to predict (e.g., `tg`, `egc`).
* **`test.csv`**: Same columns in both rounds:
  * `id`: Unique identifier for the prediction row.
  * `smiles`: SMILES representation of the polymer repeat unit.
  * `target_type`: Categorical string indicating which property needs to be predicted.
* **`PI1M.csv` (Round 2 Only)**: Contains a single column `SMILES` representing unlabeled background polymer structures.

---


## 1. Executive Summary

1. **Dataset Size Expansion**: The total number of records increased from **10,286** (Round 1) to **12,349** (Round 2), representing a **20.06% expansion**.
2. **Introduction of Multi-Task Properties**: Round 1 only contained 2 properties (`tg` and `egc`). Round 2 expands this to **7 properties**, adding `eea`, `egb`, `ei`, `eps`, and `nc`.
3. **Identical Subsets for R1 Properties**: The sizes of the `tg` (4,143 train, 2,763 test) and `egc` (2,028 train, 1,352 test) datasets are **exactly identical** across both rounds.
4. **Significant Cross-Round Leakage**: 
   > [!WARNING]
   > **55.64%** of the Round 2 Test set SMILES are present in the Round 1 Train set. Models trained on the Round 1 Train set will have pre-exposure to over half of the Round 2 evaluation targets, leading to overly optimistic validation scores if not accounted for.
5. **Train-Test SMILES Overlap in Round 2**:
   > [!NOTE]
   > In Round 2, there is an overlap of **457 SMILES** between the Train and Test sets. This is not a data leak, but rather due to the multi-task nature of the dataset where the same polymer repeat unit is measured for different target properties across train and test splits.
6. **Monomer Structural Shifts**: Round 2 polymers are, on average, slightly smaller (lower Molecular Weight and heavy atom counts) and less hydrophobic (lower LogP) than Round 1.
7. **Chemical Space Consistency**: The Tanimoto similarity within and between rounds is consistent (~0.12), indicating a diverse and structurally stable chemical space.

---

## 2. Dataset Size and Split Comparison

Both Round 1 and Round 2 maintain an exact **60.00% / 40.00%** Train/Test split ratio.

| Round | Train Records | Test Records | Total Records | Train Ratio % |
| :--- | :---: | :---: | :---: | :---: |
| **Round 1** | 6,171 | 4,115 | 10,286 | 59.99% |
| **Round 2** | 7,409 | 4,940 | 12,349 | 60.00% |
| **Change** | **+1,238** | **+825** | **+2,063** | **+20.06%** |

---

## 3. Target Type Distribution

In Round 1, only two target properties were measured. Round 2 introduces five new properties:

- `tg`: Glass transition temperature
- `egc`: Cohesive energy density
- `eea`: Electron affinity (New)
- `egb`: Band gap (New)
- `ei`: Ionization energy (New)
- `eps`: Dielectric constant (New)
- `nc`: Refractive index (New)

### Records Count by Target Type
| Round / Split | `tg` | `egc` | `eea` | `egb` | `ei` | `eps` | `nc` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1 Train** | 4,143 | 2,028 | 0 | 0 | 0 | 0 | 0 | **6,171** |
| **R1 Test** | 2,763 | 1,352 | 0 | 0 | 0 | 0 | 0 | **4,115** |
| **R2 Train** | 4,143 | 2,028 | 221 | 337 | 222 | 229 | 229 | **7,409** |
| **R2 Test** | 2,763 | 1,352 | 147 | 224 | 148 | 153 | 153 | **4,940** |

### Proportions by Target Type (%)
| Round / Split | `tg` | `egc` | `eea` | `egb` | `ei` | `eps` | `nc` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1 Train/Test** | 67.14% | 32.86% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **R2 Train/Test** | 55.92% | 27.37% | 2.98% | 4.55% | 3.00% | 3.09% | 3.09% |

> [importance]
> The absolute sizes of the `tg` and `egc` datasets are **identical** between R1 and R2. The new properties are introduced in relatively small quantities, ranging from 221 to 337 training samples each.

---

## 4. SMILES Overlap and Leakage Analysis

### Unique SMILES counts:
- **Round 1 Train**: 6,158 unique SMILES (out of 6,171 rows)
- **Round 1 Test**: 4,111 unique SMILES (out of 4,115 rows)
- **Round 2 Train**: 6,565 unique SMILES (out of 7,409 rows)
- **Round 2 Test**: 4,497 unique SMILES (out of 4,940 rows)

### Overlap Matrix (Unique SMILES)
| Dataset Split | R1 Train | R1 Test | R2 Train | R2 Test | PI1M (1M background) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **R1 Train** | **6,158** | 5 | 3,761 (61.08%) | 2,502 (40.63%) | 99 (1.61%) |
| **R1 Test** | 5 | **4,111** | 2,475 (60.20%) | 1,696 (41.26%) | 65 (1.58%) |
| **R2 Train** | 3,761 | 2,475 | **6,565** | 457 (6.96%) | 167 (2.54%) |
| **R2 Test** | 2,502 | 1,696 | 457 | **4,497** | 116 (2.58%) |

### Key Insights on Overlaps:
1. **R1 Train vs. R2 Test Leakage**:
   - **2,502** of the **4,497** unique SMILES in Round 2 Test (55.64%) are present in the Round 1 Train set.
   - If a model was trained on the complete Round 1 Train set, it has already seen the structure of 55.64% of the Round 2 test set.
2. **R2 Train vs. R2 Test Overlap**:
   - There are **457** overlapping SMILES between R2 Train and R2 Test.
   - In single-task models, this would be data leakage. In this multi-task setup, it indicates that the same polymer repeat unit has some of its properties in the Train set and other properties in the Test set.
3. **PI1M Coverage**:
   - Very few molecules (~1.6% in R1, ~2.5% in R2) are present in the PI1M background database. The competition molecules represent a highly customized chemical space, though PI1M could still serve for unsupervised representation pre-training due to its size.

---

## 5. Duplicate SMILES with Target Conflicts

There are minor duplicate records inside the training sets with the same SMILES and target property but slightly different experimental values.

- **Round 1 Train**: 6 conflicting groups (12 rows total). The largest target conflict is for `tg` where the same SMILES has target values of `262.0` and `286.0` (diff of `24.0`).
- **Round 2 Train**: 3 conflicting groups (6 rows total). The largest conflict is a `tg` diff of `10.98`.

*Recommendation: For training single-task regression models, duplicates should be averaged or resolved by keeping the median value to remove experimental noise.*

---

## 6. RDKit Molecular Descriptors Analysis

We calculated the molecular descriptors of the polymer repeat units. All SMILES are valid in RDKit.

### Mean Molecular Descriptors
| Dataset Split | Mol. Wt (MW) | LogP | Heavy Atoms | Rings | Fsp3 | Connection Points (`*`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1 Train** | 409.77 | 4.74 | 29.60 | 3.21 | 0.37 | 2.0 |
| **R1 Test** | 409.83 | 4.80 | 29.61 | 3.22 | 0.37 | 2.0 |
| **R2 Train** | 373.19 | 4.30 | 26.83 | 2.92 | 0.36 | 2.0 |
| **R2 Test** | 367.70 | 4.26 | 26.33 | 2.81 | 0.36 | 2.0 |

### Monomer Structural Shift:
- **Smaller Monomers**: Round 2 polymer repeat units are, on average, slightly smaller (MW decreases from ~410 to ~370, and heavy atoms decrease from ~29.6 to ~26.5).
- **Lower Hydrophobicity**: Average LogP decreases from ~4.75 to ~4.28, meaning Round 2 polymers are slightly less hydrophobic.
- **Connection Points**: Every single molecule in both Round 1 and Round 2 has exactly **2.0 connection points** (dummy atoms `*`), confirming that all repeating units are linear polymers.

---

## 7. Target Value Distribution Comparison

The target distributions for the overlapping target types (`tg` and `egc`) were analyzed to see if they shifted.

### Target Value Summary Statistics
| Property | Round | Count | Min | Mean | Median | Max | Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`egc`** | Round 1 | 2,028 | 0.103 | 4.531 | 4.613 | 9.863 | 1.557 |
| **`egc`** | Round 2 | 2,028 | 0.020 | 4.529 | 4.614 | 9.863 | 1.568 |
| **`tg`** | Round 1 | 4,143 | -118.000 | 140.099 | 132.000 | 490.000 | 109.386 |
| **`tg`** | Round 2 | 4,143 | -109.820 | 143.459 | 136.400 | 495.000 | 109.084 |

### Target Values for New Properties (Round 2 Only)
- **`eea`**: 221 train samples, Mean = 2.278 eV, Range = [0.394, 5.144] eV
- **`egb`**: 337 train samples, Mean = 4.276 eV, Range = [0.507, 10.114] eV
- **`ei`**: 222 train samples, Mean = 6.346 eV, Range = [4.026, 9.838] eV
- **`eps`**: 229 train samples, Mean = 4.577, Range = [2.610, 9.090]
- **`nc`**: 229 train samples, Mean = 1.934, Range = [1.560, 2.758]

### Key Insights:
- **`egc` Distribution**: The `egc` distributions are practically identical. The counts are exactly equal, and statistics are nearly identical, showing that this data subset is unchanged.
- **`tg` Distribution**: Although the counts are the same (4,143), the target values are slightly different. The mean shifted from `140.10` to `143.46`, and the median shifted from `132.0` to `136.4`. This indicates that the `tg` targets have been updated, cleaned, or slightly recalibrated in Round 2.

---

## 8. Chemical Space & Fingerprint Similarity

Morgan Fingerprints (radius=2, 2048 bits) were calculated for the train datasets to analyze chemical space overlap.

- **Tanimoto Similarity**:
  - Within Round 1: **0.1219 ± 0.0783**
  - Within Round 2: **0.1184 ± 0.0780**
  - Between Round 1 & Round 2: **0.1195 ± 0.0776**
- **Interpretation**: 
  - An average Tanimoto similarity of ~0.12 indicates that both datasets represent a highly diverse chemical space (very low structural similarity between random pairs).
  - The cross-round similarity (0.1195) is in the exact same range as the within-round similarity, meaning that Round 2 covers a structurally very similar chemical space to Round 1, despite the slightly smaller monomer sizes.
- **PCA Visualization**: The PCA projection shows that Round 1 and Round 2 are fully intermixed in chemical space. There are no major isolated islands unique to either round, indicating that models trained on Round 1 should generalize well structurally to Round 2.

---

## 9. Modeling Recommendations

1. **Leverage Cross-Round Data**: Because 60-61% of the Round 2 Train/Test SMILES overlap with Round 1 Train/Test, we can use Round 1 data to boost performance on Round 2, especially for the overlapping properties `tg` and `egc`.
2. **Handle Multi-Task Learning**: Round 2 contains 7 targets, but many SMILES only have 1 or 2 properties measured. Designing a model that can handle missing targets (e.g., mask loss for missing values or use multi-task networks) is highly recommended.
3. **Beware of Validation Leakage**:
   - Because 55.64% of the Round 2 Test set is in the Round 1 Train set, if you train a model on Round 1 Train and evaluate on Round 2 Test, you will get an optimistic evaluation score.
   - For a clean local validation setup, construct a validation split that keeps overlapping SMILES together (GroupKFold or custom split based on SMILES identity).
