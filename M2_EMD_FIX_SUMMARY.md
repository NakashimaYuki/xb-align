# M2 Phase: Sinkhorn-EMD = 0 Bug Fix - Complete Summary

**Date**: 2025-11-21
**Status**: ✅ FIXED - Two-layer solution implemented

---

## Executive Summary

Fixed the Sinkhorn-EMD = 0 bug through a **two-layer solution**:

1. **Architectural Fix (Union Support)**: Resolved the design flaw where baseline molecules had 100% OOV rate, causing zero histogram
2. **Implementation Fix (Deterministic Hash)**: Fixed the non-deterministic hash function in `SimpleEnvFeaturizer` that caused vocabulary mismatch

Both issues were critical and needed to be addressed.

---

## Problem Statement

### Observed Issue

```
Macro alignment metrics (baseline vs DrugBank):
  Sinkhorn EMD : 0.000000000000e+00  ← THE PROBLEM
  MMD²         : 0.519000
  L2 distance  : 0.354000
```

- **EMD = 0** despite distributions clearly differing (MMD² and L2 non-zero)
- This suggested either cost matrix issues, histogram normalization problems, or Sinkhorn parameters

### User's Initial Hypothesis

1. Cost matrix configuration causing many costs ≈ 0
2. Histogram/normalization issues causing accidental cancellation
3. Sinkhorn epsilon too large, "flattening" effective costs

**User's key requirement**: Systematic fix, not just dismissing as "numerical artifact"

---

## Root Cause Analysis

### Primary Cause 1: Design - Unhandled Vocabulary Mismatch

**Diagnostic findings:**

```
>>> Histogram diagnostics:
    ref.sum        = 1.000000
    gen.sum        = 0.000000   ← Baseline histogram is ALL ZEROS
    L1 diff        = 1.000000
    L2 diff        = 0.354206

>>> histogram_from_changed_atoms diagnostics:
    total changed atoms = 294
    matched to ref      = 0      ← 100% OOV rate
    out of vocabulary   = 294
```

**Issue**:
- Baseline used `histogram_from_changed_atoms()` which **discards OOV descriptors**
- All baseline `(env_id, elem)` pairs were out-of-vocabulary relative to DrugBank reference
- Result: `p = [0, 0, ..., 0]` → Sinkhorn produces `T = 0` → EMD = 0

**Why 100% OOV?**
- NP scaffold chemical environments fundamentally differ from DrugBank drugs
- Even though elements (N, O, S, etc.) overlap, their chemical environments (env_id hashes) don't

### Primary Cause 2: Implementation - Non-Deterministic Hash

**Diagnostic findings:**

```
============================================================
Checking overlap with current env_featurizer on drugs
============================================================

Overlap Analysis:
  Ref keys:        109
  Current keys:    109
  Shared keys:     0 (0.0% of ref)  ← 0% overlap despite same drug data!
```

**Issue**:
- `SimpleEnvFeaturizer.encode()` used Python's built-in `hash()` function
- Python 3.3+ has hash randomization for security
- Each Python session produces different hash values for the same input
- DrugBank reference built in one session had different env_id values than baseline computed in another session

**Evidence**:

```python
# OLD CODE (BROKEN)
def encode(self, mol, atom_idx):
    ...
    key = (sym, aromatic, degree, tuple(neigh_syms))
    return hash(key) & 0x7FFFFFFF  # Non-deterministic!
```

Running the same code twice:
- Run 1: `env_id = 380660759`
- Run 2: `env_id = 349967143`  (DIFFERENT!)

This is why we had 100% OOV even though we were looking at the same drug molecules.

---

## Solution Implemented

### Layer 1: Union Support Architecture

**Core Idea**: Instead of comparing distributions on a fixed vocabulary (ref.keys only), compare them on the **union of both vocabularies**.

**Key Changes**:

1. **New Function: `count_position_descriptors()`**
   - Counts all `(env_id, elem)` pairs in molecules **without reference restriction**
   - Returns raw counts as `Dict[PositionDescriptor, int]`

2. **New Function: `build_union_support()`**
   - Constructs union keys: `keys_all = ref.keys ∪ baseline_keys`
   - Builds histograms on union support:
     - `p_all[i]`: baseline frequency for key i (0 if i is ref-only)
     - `q_all[i]`: reference frequency for key i (0 if i is baseline-only)
   - Builds cost/kernel matrices on union support

3. **New Function: `compute_macro_metrics_union()`**
   - Computes EMD, MMD², L2 on union support
   - Ensures all baseline mass is accounted for

**Mathematical Interpretation**:

- **Old approach**: Project baseline onto ref support → loses all OOV mass → `p = 0`
- **New approach**: Extend support to include both distributions → `p` and `q` are proper distributions on common space

### Layer 2: Deterministic Hash Implementation

**Fix**: Replaced Python's `hash()` with MD5-based deterministic hash.

**Implementation**:

```python
import hashlib

def encode(self, mol, atom_idx):
    ...
    # Create string representation
    key_str = f"{sym}_{aromatic}_{degree}_{'_'.join(neigh_syms)}"

    # Use MD5 for deterministic hashing
    hash_obj = hashlib.md5(key_str.encode('utf-8'))
    hash_bytes = hash_obj.digest()[:8]
    hash_int = int.from_bytes(hash_bytes, byteorder='big')
    return hash_int & 0x7FFFFFFF
```

**Version Management**:

```python
@staticmethod
def version() -> str:
    return "simple_env_v1.1"  # Updated from v1.0
```

- Added version tracking to `SimpleEnvFeaturizer`
- Version stored in `.npz` files and validated on load
- Automatic warning/error if version mismatch detected

**Verification**:

After fix:
```
Overlap Analysis:
  Ref keys:        109
  Current keys:    109
  Shared keys:     109 (100.0% of ref)  ← PERFECT!
```

---

## Testing

### Unit Tests Added

**Test 1: `test_union_support_all_oov()`**
- Creates ref with 3 keys, baseline with 3 completely different keys
- Verifies union support has 6 keys
- Verifies EMD > 0 (not collapsed to zero)
- **Result**: ✅ PASSED

**Test 2: `test_union_support_partial_overlap()`**
- Creates ref with 4 keys, baseline with 2 shared + 2 new keys
- Verifies correct histogram construction on union support
- Verifies EMD ordering (perfect match < partial overlap)
- **Result**: ✅ PASSED

**All existing tests**: ✅ 17/17 PASSED

---

## Integration Changes

### Files Modified

1. **xb_align/core/env_featurizer.py**
   - Replaced `hash()` with deterministic MD5-based hash
   - Added version tracking (v1.0 → v1.1)

2. **xb_align/priors/macro_align.py**
   - Added `count_position_descriptors()`
   - Added `build_union_support()`
   - Added `compute_macro_metrics_union()`
   - Added version checking to `load_macro_reference()`

3. **xb_align/eval/macro_eval.py**
   - Added `count_baseline_descriptors()`

4. **xb_align/scripts/run_macro_baseline.py**
   - Changed to use union-based metrics computation
   - Added OOV statistics to output
   - Preserved histogram visualization on ref support

5. **xb_align/data/build_halopos_stats.py**
   - Added version saving to .npz output

6. **tests/test_macro_align.py**
   - Added union support test cases

### Data Files Rebuilt

- **data/processed/drug_halopos_ref.npz**
  - Rebuilt with deterministic hash
  - Now includes `env_version = "simple_env_v1.1"`
  - 109 unique `(env_id, elem)` pairs

---

## Expected Results After Fix

### With Union Support + Deterministic Hash

```
Macro alignment metrics (baseline vs DrugBank, union support):
  Sinkhorn EMD : >0   (likely 0.5-1.0 range)
  MMD²         : >0   (unchanged, still valid)
  L2 distance  : >0   (unchanged, still valid)

Union support statistics:
  ref keys         = 109
  baseline-only    = X (depends on NP scaffold diversity)
  union keys       = 109 + X
  shared keys      = Y (depends on overlap)
  OOV rate         = (X / (Y+X)) * 100%
```

### Interpretation

- **EMD > 0**: Now properly quantifies transport cost between distributions
- **Baseline-only keys**: Represents NP-specific chemical environments
- **OOV rate**: Shows how different NP scaffolds are from DrugBank
  - High OOV (e.g., 80%+): NP scaffolds very different from drugs
  - Low OOV (e.g., <20%): NP scaffolds similar to drugs

---

## Key Insights

### Design Level

1. **Vocabulary mismatch is a feature, not a bug**: NP scaffolds SHOULD be different from DrugBank initially. The goal of GFlowNet is to bridge this gap.

2. **Union support is conceptually correct**: When comparing distributions with non-overlapping support, you must use union support or a continuous embedding space.

3. **OOV tracking is valuable**: The OOV rate is itself a useful metric for diversity/novelty.

### Implementation Level

1. **Python's `hash()` is not deterministic**: Critical bug for any reproducibility-sensitive application.

2. **Version management is essential**: Without version tracking, subtle implementation changes can cause silent failures.

3. **Test non-overlapping distributions**: Edge cases like 100% OOV reveal design flaws.

---

## Comparison: Before vs After

| Aspect | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| **Hash Function** | Python `hash()` (non-deterministic) | MD5-based (deterministic) |
| **Version Tracking** | None | `simple_env_v1.1` in code + .npz files |
| **Vocabulary Overlap** | 0% (due to hash issue) | 100% (for DrugBank self-comparison) |
| **Baseline OOV Handling** | Discarded → zero histogram | Counted in union support |
| **EMD Value** | 0.000000 (collapsed) | >0 (meaningful) |
| **Support Space** | ref.keys only (109 dims) | ref.keys ∪ baseline.keys (109+ dims) |

---

## Future Recommendations

1. **M3 Integration**:
   - GFlowNet reward should use `compute_macro_metrics_union()`
   - Track OOV rate over training (should decrease)
   - Consider separate metrics for "coverage" (shared keys) and "divergence" (EMD on shared)

2. **Alternative Approaches** (if needed):
   - Use continuous embeddings (e.g., learned atom environment vectors) instead of discrete env_id
   - Use kernel methods (MMD) as primary metric, EMD as secondary

3. **Monitoring**:
   - Always check OOV rate in diagnostics
   - Alert if overlap drops below expected threshold
   - Track version consistency in automated pipelines

---

## Lessons Learned

1. **Always use deterministic functions in ML pipelines**: Non-deterministic behavior breaks reproducibility and can cause silent failures.

2. **Design for vocabulary mismatch from the start**: In molecular generation, comparing novel vs reference distributions is the norm, not the exception.

3. **Version everything that affects data**: Not just models, but also featurizers, preprocessors, and any code that creates cached data.

4. **Test edge cases**: 100% OOV, 0% overlap, etc. These reveal design assumptions.

5. **User intuition is valuable**: The user's insistence that "EMD=0 is unreasonable" led to uncovering deep issues.

---

## Status: READY FOR BASELINE PIPELINE RUN

Next step: Run the full baseline pipeline to verify EMD > 0 in the real scenario.

```bash
python xb_align/scripts/run_macro_baseline.py \
    --np-scaffolds data/processed/np_scaffolds.parquet \
    --halopos-ref data/processed/drug_halopos_ref.npz \
    --graph-mlm models/graph_mlm.pt \
    --envfrag-table data/processed/envfrag_table.npz \
    --out-dir outputs/baseline_m2_fixed \
    --n-samples 5000 \
    --seed 42
```
