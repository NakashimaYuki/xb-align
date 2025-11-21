# XB-Align: GFlowNet-based Halogen/Heteroatom Doping for Natural Product Molecules

A research project for learning and generating halogen/heteroatom position distributions in natural product-like drug molecules using GFlowNet.

## Project Overview

This project aims to:

1. **Learn Position Distributions**: Extract halogen (F, Cl, Br, I) and heteroatom (N, O, S, P) position distributions from drug databases (DrugCentral/ChEMBL)
2. **Generate NP-like Molecules**: Apply learned distributions to natural product (NP) scaffolds from CNPD-ETCM to generate drug-like molecules
3. **Multi-objective Optimization**: Balance position distribution alignment with ADMET properties, halogen bond geometry, and synthetic accessibility

## Project Structure

```
xb_align_v4/
├── data/
│   ├── raw/                          # Raw input data
│   │   ├── CNPD_ETCM_merged.xlsx    # NP molecules (rename your file to this)
│   │   └── drugs_raw.csv             # Drug molecules with columns: drug_id, name, smiles
│   └── processed/                    # Generated data files
│       ├── np_scaffolds.parquet      # Extracted NP scaffolds
│       ├── drugs_std.parquet         # Standardized drug data
│       ├── drug_halopos_ref.npz      # Position reference distribution
│       ├── envfrag_table.npz         # Env x Fragment co-occurrence table
│       └── graph_mlm.pt              # Trained Graph-MLM model
├── xb_align/
│   ├── core/                         # Core utilities
│   │   └── env_featurizer.py        # Environment encoding
│   ├── data/                         # Data processing
│   │   ├── prepare_np_scaffolds.py
│   │   ├── prepare_drugs.py
│   │   ├── build_halopos_stats.py
│   │   └── build_envfrag_table.py
│   ├── priors/                       # Prior models
│   │   ├── atom_vocab.py
│   │   ├── position_descriptor.py
│   │   ├── pas_energy.py
│   │   ├── envfrag_energy.py
│   │   ├── graph_mlm.py
│   │   ├── graph_mlm_data.py
│   │   ├── train_graph_mlm.py
│   │   └── macro_align.py           # M2: Macro alignment (EMD, MMD)
│   ├── rewards/                      # Reward functions
│   │   └── prior_micro.py
│   ├── baseline/                     # M2: Baseline generation
│   │   ├── random_doping.py         # Random atom substitutions
│   │   ├── generator.py             # Batch sampling
│   │   └── scoring.py               # Prior-based ranking
│   ├── eval/                         # M2: Evaluation tools
│   │   └── macro_eval.py            # Histogram computation and plotting
│   ├── scripts/                      # Analysis scripts
│   │   ├── compare_prior_on_drugs_vs_random.py
│   │   └── run_macro_baseline.py    # M2: Baseline evaluation
│   └── gfn/                          # GFlowNet (future implementation)
└── tests/                            # Unit tests (56 tests total)
    ├── test_data_pipeline.py        # M1 tests
    ├── test_macro_align.py          # M2: Macro alignment tests
    ├── test_random_doping.py        # M2: Random doping tests
    ├── test_baseline_generator.py   # M2: Generator tests
    ├── test_baseline_scoring.py     # M2: Scoring tests
    └── test_macro_eval.py           # M2: Evaluation tests
```

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n xb_align python=3.10 -y
conda activate xb_align
```

### 2. Install Dependencies

```bash
# Core scientific computing libraries
pip install "numpy<2.0" pandas pyarrow openpyxl

# RDKit (chemistry toolkit)
conda install -c conda-forge rdkit -y

# PyTorch (CPU version for Windows)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric (graph neural networks)
pip install torch-geometric

# Additional utilities
pip install scikit-learn tqdm pyyaml pytest matplotlib
```

For **Linux with GPU**, replace PyTorch installation with:
```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Prepare Input Data

Place your input data files in `data/raw/`:

- **CNPD-ETCM-merged.xlsx**: Excel file with columns `ID`, `Name`, `Smiles` (natural products)
  - The script automatically detects Chinese filenames
  - Current implementation processes the real CNPD-ETCM dataset (74,278 rows)
- **drugbank_data_cleaned.csv**: CSV file with column `SMILES` or `smiles` (drug molecules)
  - This file is converted to drugs_raw.csv format by prepare_drugbank_raw.py
  - Current implementation uses DrugBank dataset (8,773 molecules, 7,809 valid after filtering)

## Usage

### M1: Data Preparation and Prior Learning

Follow these steps in order:

#### Step 1: Extract NP Scaffolds

```bash
python -m xb_align.data.prepare_np_scaffolds
```

Output: `data/processed/np_scaffolds.parquet` (27,552 unique scaffolds from 74,278 molecules)

#### Step 2: Convert DrugBank to Standard Format

```bash
python -m xb_align.data.prepare_drugbank_raw
```

Output: `data/raw/drugs_raw.csv` (8,773 molecules in standardized format)

#### Step 3: Standardize Drug Data

```bash
python -m xb_align.data.prepare_drugs
```

Output: `data/processed/drugs_std.parquet` (7,809 valid molecules after filtering)

#### Step 4: Build Position Reference Distribution

```bash
python -m xb_align.data.build_halopos_stats
```

Output: `data/processed/drug_halopos_ref.npz` (109 unique position-element pairs, includes version info)

#### Step 5: Build Env×Frag Co-occurrence Table

```bash
python -m xb_align.data.build_envfrag_table
```

Output: `data/processed/envfrag_table.npz` (109 environment-element pairs with log probabilities, includes version info)

#### Step 6: Train Graph-MLM Model

```bash
python -m xb_align.priors.train_graph_mlm
```

Output: `data/processed/graph_mlm.pt` (trained on 7,809 DrugBank molecules)

Training parameters (can be modified in the script):
- max_mols: 50000 (actual: 7,809 molecules used)
- batch_size: 64
- num_epochs: 5
- learning_rate: 1e-3

Training results: Loss improved from 0.6984 (epoch 1) to 0.5462 (epoch 5)

#### Step 7: Verify Version Consistency (Optional but Recommended)

Check that env_featurizer and data files are compatible:

```bash
# Check drug_halopos_ref.npz version
python check_env_version.py

# Check envfrag_table.npz version
python check_envfrag_version.py
```

These scripts verify:
- Version matches between code and data files
- 100% overlap of env_id values (confirms deterministic hash)
- Automatic warnings if rebuild is needed

#### Step 8: Evaluate Prior Model

Compare real drugs vs randomly perturbed molecules at the same positions:

```bash
python -m xb_align.scripts.compare_prior_on_drugs_vs_random
```

This script performs fair evaluation by:
1. Selecting k=5 random positions in each molecule
2. Creating perturbed version with different atoms at those positions
3. Comparing prior scores at the SAME positions for both real and fake

**Validation Results (v0.1.2):**
- Mean(delta = real - fake): 33.282 (significantly > 0) ✓
- Fraction(delta > 0): 100% (far exceeds 50% threshold) ✓
- Real drugs score: -53.360, Fake score: -86.642

**Conclusion:** The model successfully learned meaningful position preferences from DrugBank data.

---

### M2: Macro Alignment and Baseline Generation

M2 implements macro-level position distribution alignment using optimal transport metrics and establishes a baseline generation system for comparison.

#### Run Baseline Generation and Evaluation

```bash
python -m xb_align.scripts.run_macro_baseline \
  --np-scaffolds data/processed/np_scaffolds.parquet \
  --halopos-ref data/processed/drug_halopos_ref.npz \
  --graph-mlm data/processed/graph_mlm.pt \
  --envfrag-table data/processed/envfrag_table.npz \
  --out-dir outputs/m2_baseline \
  --n-samples 5000 \
  --max-changes 5 \
  --seed 42
```

**Output files:**
- `outputs/m2_baseline/baseline_top_samples.csv`: Top 2000 ranked baseline molecules
- `outputs/m2_baseline/macro_hist_baseline_vs_drugbank.png`: Distribution comparison plot
- `outputs/m2_baseline/metrics.txt`: Macro alignment metrics (including OOV rate and overlap statistics)

**Actual metrics (baseline vs DrugBank on union support):**
```
Sinkhorn EMD  : 4.520276955164e-01
MMD^2         : 0.054863
L2 distance   : 0.322856
OOV rate      : 62.80%
Shared keys   : 61 / 109 (56% of DrugBank reference)
```

These metrics are computed on **union support space** (ref keys ∪ baseline keys) to properly handle vocabulary mismatch between NP scaffolds and DrugBank.

#### M2 Components

**Macro Alignment Module** (`xb_align.priors.macro_align`):
- **MacroAlignReference**: Loads reference distribution from `drug_halopos_ref.npz` with version validation
- **Sinkhorn-EMD**: Entropic-regularized optimal transport distance
- **MMD²**: Maximum Mean Discrepancy with RBF kernel
- **Union Support**: `build_union_support()` creates ref ∪ baseline vocabulary to handle OOV correctly
- **`compute_macro_metrics_union()`**: Computes metrics on union support, preventing histogram collapse
- **Histogram builders**: For complete molecules or changed-atom subsets

**Baseline Generator** (`xb_align.baseline`):
- **random_doping**: Random single-atom substitutions (C→N/O/S/P or C-H→C-F/Cl/Br/I)
- **generator**: Batch sampling from NP scaffold library
- **scoring**: Prior-based ranking using PriorMicroScorer

**Evaluation Framework** (`xb_align.eval.macro_eval`):
- Histogram computation from baseline samples
- Visualization: Side-by-side distribution comparison plots

#### Key Metrics

1. **Sinkhorn-EMD**: Measures cost of transforming baseline distribution to DrugBank distribution
   - Lower is better (0 = perfect match)
   - Accounts for position similarity via cost matrix

2. **MMD²**: Kernel-based distribution distance
   - Lower is better (0 = perfect match)
   - Uses RBF kernel with bandwidth=1.0

3. **L2 Distance**: Simple Euclidean distance between histograms
   - Lower is better (0 = perfect match)

---

### Run Tests

```bash
pytest
```

Current test coverage:
- **M1 tests**: 11 tests (data pipeline, priors, Graph-MLM)
- **M2 tests**: 17 tests in test_macro_align.py (includes union support tests)
- **Total**: All tests passing ✅

## Key Concepts

### Position Descriptor

A `PositionDescriptor` captures the local environment and element at a specific atom position:
- `env_id`: **Deterministic** MD5-based encoding of local atom environment (atom type, aromaticity, degree, neighbor types)
- `elem`: Element symbol (F, Cl, Br, I, N, O, S, P)

**Version tracking**: Both `SimpleEnvFeaturizer` (v1.1) and data files include version information to ensure compatibility.

### Prior Components

1. **Graph-MLM**: Graph neural network that predicts masked atom types based on molecular context
2. **EnvFrag Energy**: Log-probability table for element occurrence in specific environments
3. **PAS Energy**: Position-specific aromatic substitution preferences (placeholder in M1)

### Combined Prior Score

```python
log_prior_micro = -(alpha * Graph_MLM_NLL) + beta * PAS_score + gamma * EnvFrag_score
```

Higher scores indicate better alignment with drug-like position distributions.

## Development Roadmap

- [x] **M1: Data preparation and prior learning (COMPLETE & VALIDATED)**
  - [x] NP scaffolds from real CNPD-ETCM data (27,552 scaffolds)
  - [x] Drug standardization from DrugBank (7,809 molecules)
  - [x] Position priors and Env×Frag table with version tracking
  - [x] Graph-MLM training with corrected masking
  - [x] Prior validation: Real > Fake (Mean delta: 33.28, 100% success rate)
  - [x] Deterministic MD5-based env_id hash (v1.1)
- [x] **M2: Macro alignment and baseline generation (COMPLETE & VALIDATED)**
  - [x] Sinkhorn-EMD and MMD² implementation for distribution comparison
  - [x] **Union support architecture** to handle vocabulary mismatch
  - [x] Random doping baseline with prior-based ranking
  - [x] Macro histogram computation and visualization
  - [x] End-to-end baseline evaluation script
  - [x] **Fixed EMD=0 bug** (dual-layer solution: design + implementation)
  - [x] **Version management system** for env_featurizer and data files
  - [x] 17 comprehensive tests including union support scenarios
  - [x] **Verified metrics**: EMD=0.452, MMD²=0.055, OOV=62.8%
- [ ] M3: GFlowNet implementation for scaffold doping
- [ ] M4: Multi-objective reward design (ADMET, halogen bonds, synthesis)
- [ ] M5: Large-scale generation and evaluation
- [ ] M6: Docking validation and case studies

## Contributing

This is a research project. For questions or collaboration:
1. Check existing issues
2. Create detailed bug reports with reproducible examples
3. Include environment information (OS, Python version, package versions)

## License

MIT License (or specify your license)

## Citation

If you use this code in your research, please cite:

```
[Your paper citation will go here]
```

## Acknowledgments

- CNPD and ETCM databases for natural product data
- DrugCentral/ChEMBL for drug molecule data
- RDKit for cheminformatics functionality
- PyTorch Geometric for graph neural network support
