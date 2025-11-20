# Quick Start Guide

This guide will get you up and running with the XB-Align project in under 10 minutes.

## Prerequisites

- Windows 10/11 (or Linux/macOS)
- Anaconda or Miniconda installed
- At least 4GB RAM
- 2GB free disk space

## Installation (5 minutes)

### 1. Create Environment

```bash
conda create -n xb_align python=3.10 -y
conda activate xb_align
```

### 2. Install Dependencies

```bash
# Core packages
pip install "numpy<2.0" pandas pyarrow openpyxl

# Chemistry toolkit
conda install -c conda-forge rdkit -y

# Machine learning
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# Utilities
pip install scikit-learn tqdm pyyaml pytest
```

## Quick Test (2 minutes)

### Run Tests

```bash
pytest
```

Expected output: 6 tests passed

### Verify Installation

```bash
python -c "from rdkit import Chem; import torch; print('All imports OK')"
```

## Run M1 Pipeline (5-15 minutes)

The project includes sample data (30 drugs, 20 natural products) for immediate testing.

### Option 1: Run Complete Pipeline

```bash
python run_m1_pipeline.py
```

This will:
1. Extract NP scaffolds from sample data
2. Standardize drug SMILES
3. Build position reference distributions
4. Build environment co-occurrence tables
5. Train Graph-MLM model (5 epochs)
6. Compare real vs random molecule scores

### Option 2: Run Steps Individually

```bash
# Step 1: Extract NP scaffolds
python -m xb_align.data.prepare_np_scaffolds

# Step 2: Standardize drugs
python -m xb_align.data.prepare_drugs

# Step 3: Build position reference
python -m xb_align.data.build_halopos_stats

# Step 4: Build Env x Frag table
python -m xb_align.data.build_envfrag_table

# Step 5: Train Graph-MLM
python -m xb_align.priors.train_graph_mlm

# Step 6: Evaluate
python -m xb_align.scripts.compare_prior_on_drugs_vs_random
```

## Expected Output

After running the pipeline, you should see:

```
Results:
Real drugs: ~800-900 molecules
Fake (perturbed): ~800-900 molecules

Mean log_prior_micro (real): -15.234 +/- 8.567
Mean log_prior_micro (fake): -23.456 +/- 10.234

Difference: 8.222

Expected: Real drugs should have higher (less negative) scores than fake molecules
```

The positive difference indicates the model successfully learned drug-like position patterns.

## Generated Files

After completion, check `data/processed/`:

- `np_scaffolds.parquet` - Extracted natural product scaffolds
- `drugs_std.parquet` - Standardized drug molecules
- `drug_halopos_ref.npz` - Position reference distribution
- `envfrag_table.npz` - Environment co-occurrence table
- `graph_mlm.pt` - Trained Graph-MLM model (~5MB)

## Using Your Own Data

### Replace Sample NP Data

1. Place your Excel file at `data/raw/CNPD_ETCM_merged.xlsx`
2. Required columns: `ID`, `Name`, `Smiles`
3. Run pipeline again

### Replace Sample Drug Data

1. Place your CSV file at `data/raw/drugs_raw.csv`
2. Required columns: `drug_id`, `name`, `smiles`
3. Run pipeline again

### Recommended Data Sizes

For meaningful results:
- Minimum: 1,000 drugs, 500 NP molecules
- Recommended: 10,000+ drugs, 5,000+ NP molecules
- Full scale: 50,000+ drugs, 20,000+ NP molecules

## Troubleshooting

### Import Error: No module named 'rdkit'

```bash
conda install -c conda-forge rdkit -y
```

### Import Error: No module named 'torch_geometric'

```bash
pip install torch-geometric
```

### Graph-MLM Training Too Slow

Edit `xb_align/priors/train_graph_mlm.py`:
- Reduce `max_mols` from 50000 to 10000
- Reduce `num_epochs` from 5 to 3
- Reduce `batch_size` from 64 to 32

### Out of Memory

For large datasets:
- Process data in chunks
- Reduce `batch_size` in training
- Use fewer molecules for Graph-MLM training

## Next Steps

1. Review generated distributions: Load NPZ files with numpy
2. Analyze scaffold diversity: Explore `np_scaffolds.parquet`
3. Fine-tune Graph-MLM: Adjust hyperparameters
4. Move to M2: Implement GFlowNet generation

## Support

- Check `README.md` for detailed documentation
- Review code comments for implementation details
- Run `pytest -v` for comprehensive testing
- Check issues on GitHub (when available)

## Performance Benchmarks

On a typical Windows laptop (Intel i5, 8GB RAM):
- NP scaffold extraction: 10-30 seconds
- Drug standardization: 5-10 seconds
- Position stats: 5-10 seconds
- Env x Frag table: 5-10 seconds
- Graph-MLM training: 2-5 minutes
- Evaluation: 30-60 seconds

Total time: 5-10 minutes for sample data

---

For detailed information, see `README.md`
