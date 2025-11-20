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
│   │   └── train_graph_mlm.py
│   ├── rewards/                      # Reward functions
│   │   └── prior_micro.py
│   ├── scripts/                      # Analysis scripts
│   │   └── compare_prior_on_drugs_vs_random.py
│   └── gfn/                          # GFlowNet (future implementation)
└── tests/                            # Unit tests
    └── test_data_pipeline.py
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
pip install scikit-learn tqdm pyyaml pytest
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

Output: `data/processed/drug_halopos_ref.npz` (109 unique position-element pairs)

#### Step 5: Build Env×Frag Co-occurrence Table

```bash
python -m xb_align.data.build_envfrag_table
```

Output: `data/processed/envfrag_table.npz` (109 environment-element pairs with log probabilities)

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

#### Step 7: Evaluate Prior Model

Compare real drugs vs randomly perturbed molecules:

```bash
python -m xb_align.scripts.compare_prior_on_drugs_vs_random
```

This script evaluates the prior scorer on real DrugBank molecules vs chemically perturbed versions, validating that the model has learned meaningful position preferences.

### Run Tests

```bash
pytest
```

## Key Concepts

### Position Descriptor

A `PositionDescriptor` captures the local environment and element at a specific atom position:
- `env_id`: Hash-based encoding of local atom environment (atom type, aromaticity, degree, neighbor types)
- `elem`: Element symbol (F, Cl, Br, I, N, O, S, P)

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

- [x] M1: Data preparation and prior learning
- [ ] M2: GFlowNet implementation for scaffold doping
- [ ] M3: Multi-objective reward design (ADMET, halogen bonds, synthesis)
- [ ] M4: Large-scale generation and evaluation
- [ ] M5: Docking validation and case studies

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
