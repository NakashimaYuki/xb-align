# Changelog

All notable changes to the XB-Align project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2025-11-21

### Fixed - M2 EMD=0 Bug: Dual-Layer Solution

**Critical Bug: Sinkhorn-EMD returning 0.000000**

**Problem Analysis**:
- **Symptom**: EMD = 0 despite MMD² = 0.519 and L2 = 0.354 showing clear distribution differences
- **Root Cause 1 (Design)**: Vocabulary mismatch - baseline had 100% OOV rate, causing histogram = all zeros
- **Root Cause 2 (Implementation)**: Non-deterministic Python `hash()` caused env_id incompatibility between runs

**Solution 1: Union Support Architecture**
- Added `count_position_descriptors()`: Counts all descriptors without reference restriction
- Added `build_union_support()`: Constructs ref ∪ baseline vocabulary space
- Added `compute_macro_metrics_union()`: Computes EMD/MMD²/L2 on union support
- **Result**: EMD changed from 0.000000 → 0.452 (meaningful value) ✅

**Solution 2: Deterministic Hash + Version Management**
- Fixed `SimpleEnvFeaturizer.encode()`: Replaced Python `hash()` with MD5-based deterministic hash
- Version: simple_env_v1.0 → **v1.1** (breaking change, requires data rebuild)
- Added version tracking to `SimpleEnvFeaturizer`, `EnvFragEnergy`, and all .npz files
- Added version validation in `load_macro_reference()` and `EnvFragEnergy.load()`
- **Result**: env_id overlap from 0% → 100% ✅

**New Version Management Tools**:
- `check_env_version.py`: Validates drug_halopos_ref.npz version and overlap
- `check_envfrag_version.py`: Validates envfrag_table.npz version and overlap

**Data Files Rebuilt**:
- `drug_halopos_ref.npz`: Rebuilt with simple_env_v1.1 (100% overlap confirmed)
- `envfrag_table.npz`: Rebuilt with envfrag_simple_env_v1.1 (100% overlap confirmed)

**Test Enhancements** (+2 tests, 17 total in test_macro_align.py):
- `test_union_support_all_oov`: Validates EMD > 0 with 100% OOV scenario
- `test_union_support_partial_overlap`: Validates EMD ordering with mixed vocabulary

**Validation Results** (Real Baseline Pipeline):
```
Union Support Metrics (baseline vs DrugBank):
  Sinkhorn EMD : 4.520276955164e-01  ✅ No longer 0!
  MMD²         : 0.054863            ✅ Reasonable value
  L2 distance  : 0.322856            ✅ Consistent

Union Support Diagnostics:
  ref keys         = 109
  baseline-only    = 103
  union keys       = 212
  shared keys      = 61 (56% of ref)
  OOV rate         = 62.80%
```

**Technical Details**:
- Union support properly handles non-overlapping vocabularies in OT
- NP scaffolds have 62.8% unique chemical environments vs DrugBank (expected)
- MD5 hash ensures reproducibility across Python sessions
- Version mismatch raises ValueError with clear rebuild instructions
- Old .npz files without version trigger UserWarning

**Documentation**:
- `M2_EMD_FIX_SUMMARY.md`: Complete technical analysis of the bug and fix
- Updated README.md with union support explanation and version management
- All changes fully documented with rationale

**Impact**:
- **M2 COMPLETE**: All original goals achieved + major infrastructure improvements
- EMD now correctly captures baseline vs DrugBank distribution difference
- Version system prevents future silent failures from featurizer changes
- Baseline metrics establish clear target for M3 GFlowNet (EMD < 0.452)

---

## [0.2.1] - 2025-11-21

### Fixed - M2 Polish and Validation

**Critical Bugs Fixed**:
- PriorMicroScorer.from_files(): Corrected GraphMLM parameters to match training
  - Was: `GraphMLM(hidden_dim=128, num_layers=3, num_atom_classes=...)`  ❌
  - Now: `GraphMLM(num_atom_types=NUM_ATOM_CLASSES, hidden_dim=128)` ✅
- EnvFragEnergy: Added missing `load()` classmethod for NPZ file loading
- sample_baseline(): Added max_attempts safety guard to prevent infinite loops (default: n_samples * 10)

**Test Enhancements** (+2 tests, 47 total):
- test_sinkhorn_simple_2d_ordering: Validates EMD with synthetic distributions
- test_histogram_consistency_full_vs_changed: Ensures histogram method consistency

**Infrastructure**:
- Created analyze_macro_distributions.py: Unified multi-source comparison script
- Documented GraphMLM configuration in docs/GRAPH_MLM_CONFIG.md
- Enhanced PriorMicroScorer.log_prior_micro() docstring with changed_atoms semantics

**Validation** (Real Data Results):
- Successfully ran baseline pipeline with 2000 samples
- Macro metrics obtained: MMD² = 0.519, L2 = 0.354 (baseline vs DrugBank)
- Created M2_VALIDATION_REPORT.md with comprehensive analysis
- All 47 M2 tests passing

---

## [0.2.0] - 2025-11-21

### Added - M2 Macro Alignment and Baseline Generation

**Core Macro Alignment Module** (`xb_align.priors.macro_align`):
- MacroAlignReference: Reference distribution over (env_id, element) position descriptors
- Sinkhorn-EMD implementation: Entropic-regularized optimal transport distance
- MMD2 implementation: Maximum Mean Discrepancy with RBF kernel
- Cost matrix builder with configurable env/element mismatch penalties
- Histogram builders for complete molecules and changed-atom subsets
- Complete metric suite (EMD, MMD2, L2) for distribution comparison

**Baseline Generator** (`xb_align.baseline`):
- random_doping.py: Random single-atom substitution on scaffolds with RDKit sanitization
- generator.py: Batch sampling from NP scaffold library (load, sample, BaselineSample dataclass)
- scoring.py: BaselinePriorRanker for prior-based ranking using PriorMicroScorer

**Evaluation Framework** (`xb_align.eval`):
- macro_eval.py: Histogram computation from baseline samples
- Visualization: Side-by-side bar plots comparing baseline vs DrugBank distributions
- Support for top-k filtering to focus on most frequent position descriptors

**CLI Script** (`xb_align/scripts/run_macro_baseline.py`):
- End-to-end baseline generation and evaluation pipeline
- Configurable parameters: n_samples, max_changes, seed
- Automatic output: metrics.txt, baseline_top_samples.csv, histogram plots
- Progress reporting with 7-step workflow

**Enhanced PriorMicroScorer**:
- Added from_files() classmethod for convenient loading from checkpoints
- Unified interface for loading Graph-MLM and EnvFrag models

### Testing
- test_macro_align.py: 13 comprehensive tests for alignment functions
  - Cost matrix construction and normalization
  - RBF kernel construction
  - Sinkhorn-EMD correctness and convergence
  - MMD2 properties and edge cases
  - Integration test with real drug_halopos_ref.npz

- test_random_doping.py: 8 tests for baseline doping
  - Deterministic behavior with fixed seeds
  - Chemical validity via RDKit sanitization
  - Respects max_changes constraint
  - Handles invalid SMILES gracefully

- test_baseline_generator.py: 9 tests for baseline sampling
  - Sample generation from scaffold library
  - Deterministic sampling with seeds
  - Molecule validity and changed_atoms consistency
  - Error handling for missing columns

- test_baseline_scoring.py: 8 tests for prior-based ranking
  - Mock scorer for isolated ranking logic
  - Invalid molecule filtering
  - Deterministic ranking
  - Edge cases (empty list, single sample)

- test_macro_eval.py: 7 tests for evaluation functions
  - Histogram computation from baseline samples
  - Plot generation and file I/O
  - Parent directory creation
  - Invalid input handling

**Total: 45 new tests, all passing**

### Performance
- Sinkhorn-EMD converges in <200 iterations with epsilon=0.1
- Cost matrix and kernel matrix precomputed once per reference
- Histogram computation scales linearly with molecule count
- Baseline generation: ~1000 molecules per minute (CPU, single-threaded)

### Technical Details
- Entropic regularization (epsilon=0.1) balances accuracy and speed in Sinkhorn
- RBF kernel bandwidth=1.0 for MMD distance metric
- Default baseline parameters: max_changes=5, top_k=2000 samples
- All modules follow English-only, no-emoji coding standards
- Comprehensive docstrings and type hints throughout

---

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
