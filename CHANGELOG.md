# Changelog

All notable changes to the XB-Align project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-20

### Added
- Initial project structure with complete M1 implementation
- Core modules: environment featurizer, position descriptors
- Data processing pipeline for NP scaffolds and drug standardization
- Prior models: Graph-MLM (GIN-based), EnvFrag energy, PAS energy (placeholder)
- Position reference distribution builder (drug_halopos_ref.npz)
- Combined prior scorer integrating Graph-MLM + EnvFrag + PAS
- Comparison script for prior validation (real vs random molecules)
- Sample data: 30 drugs + 20 natural products for testing
- Comprehensive documentation (README, QUICKSTART, PROJECT_STATUS)
- Environment setup scripts (Windows + Linux/Mac)
- One-command M1 pipeline runner
- Unit tests with pytest (6 tests, all passing)
- Git repository with clean commit history

### Changed
- Enhanced NP scaffold extraction to handle real CNPD-ETCM data (74K+ rows)
- Added explicit filtering for 'SDF' placeholders in SMILES data
- Improved progress reporting with detailed statistics
- Support for Chinese filenames with automatic detection

### Fixed
- Git push conflict with remote LICENSE file (resolved via merge)
- SMILES validation to filter invalid 'SDF' placeholders

### Performance
- Successfully processed 74,278 NP molecules → 27,552 unique scaffolds (86.9% valid)
- Efficient handling of large datasets with progress tracking every 10k rows

### Technical Details
- Python 3.10, RDKit, PyTorch 2.3.1, PyTorch Geometric
- Parquet format for efficient data storage
- NPZ format for statistical distributions
- PyTorch checkpoints for trained models

---

## Release Notes

### v0.1.0 - M1 Complete
This release includes all M1 deliverables:
1. ✅ np_scaffolds.parquet (27,552 unique scaffolds)
2. ✅ drugs_std.parquet (standardized drug data)
3. ✅ drug_halopos_ref.npz (position reference)
4. ✅ envfrag_table.npz (environment co-occurrence)
5. ✅ graph_mlm.pt (trained Graph-MLM model)
6. ✅ Validation scripts and comparison tools

**Ready for M2: GFlowNet implementation**
