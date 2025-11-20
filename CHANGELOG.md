# Changelog

All notable changes to the XB-Align project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-11-20

### Changed
- **MAJOR FIX**: Refactored prior comparison script to ensure fair evaluation
- compare_prior_on_drugs_vs_random.py now compares real vs fake at SAME k=5 positions
- Implemented perturb_at_positions() for targeted position perturbation
- Replaced separate changed_atoms definitions with unified position-based comparison

### Validation
- **M1 VALIDATION COMPLETE**: Prior model successfully validated
- Mean(delta = real - fake): 33.282 (significantly > 0)
- Fraction(delta > 0): 100% (far exceeds 50% threshold)
- Real drugs score: -53.360, Fake score: -86.642
- Conclusion: Graph-MLM + EnvFrag learned meaningful position preferences from DrugBank

### Technical Details
- Fair comparison uses same k positions for both real and perturbed molecules
- Eliminates bias from unequal position counts in previous implementation
- Statistical significance confirmed across 36 valid comparison pairs

---

## [0.1.1] - 2025-11-20

### Added
- Data pipeline to ingest DrugBank (drugbank_data_cleaned.csv) and build standardized drugs_raw.csv
- New module: prepare_drugbank_raw.py for converting DrugBank CSV to standard format
- Reference distributions (drug_halopos_ref.npz and envfrag_table.npz) based on full DrugBank dataset (7809 molecules)
- Comprehensive test suite for M1 artifacts (test_priors_micro.py with 5 tests)
- Tests for prior scorer, envfrag table, Graph-MLM checkpoint, and drugs_std.parquet structure

### Changed
- Updated Graph-MLM data pipeline to use explicit node masks (batch.mask) for MLM loss computation
- Replaced complex batch mask handling with simple boolean mask approach in train_graph_mlm.py
- Simplified graph_mlm_data.py to return Data objects with explicit mask field
- compare_prior_on_drugs_vs_random.py now evaluates priors on full DrugBank dataset instead of sample data
- All prior statistics now based on real DrugBank molecules (7809 drugs with 2314 containing halogens)

### Fixed
- Corrected Graph-MLM mask implementation to avoid offset errors during batch processing
- Ensured MLM loss is computed only on masked node positions using PyG batch.mask
- Fixed potential issues with mask_indices concatenation in batched graphs

### Performance
- Trained Graph-MLM on 7809 DrugBank molecules over 5 epochs
- Training loss improved from 0.6984 to 0.5462
- Generated 109 unique (env_id, elem) pairs for position priors
- All 11 tests passing (6 original + 5 new M1 tests)

### Technical Details
- drugs_std.parquet: 7809 molecules (89% valid rate from 8773 raw SMILES)
- drug_halopos_ref.npz: 109 position-element pairs with normalized frequencies
- envfrag_table.npz: 109 environment-element pairs with log probabilities
- graph_mlm.pt: GIN-based model trained on full DrugBank with corrected masking

---

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
