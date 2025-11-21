# tests/test_macro_align.py
"""Tests for macro alignment module."""

import numpy as np
import pytest
from pathlib import Path

from xb_align.priors.macro_align import (
    _build_cost_matrix,
    _build_rbf_kernel,
    sinkhorn_emd,
    mmd2,
    load_macro_reference,
    compute_macro_metrics,
    build_union_support,
    compute_macro_metrics_union,
    MacroAlignReference,
)
from xb_align.priors.position_descriptor import PositionDescriptor


def test_build_cost_matrix():
    """Test cost matrix construction."""
    keys = [
        PositionDescriptor(env_id=1, elem="F"),
        PositionDescriptor(env_id=1, elem="Cl"),
        PositionDescriptor(env_id=2, elem="F"),
        PositionDescriptor(env_id=2, elem="Cl"),
    ]

    C = _build_cost_matrix(keys, env_mismatch_cost=2.0, elem_mismatch_cost=1.0, normalize=False)

    assert C.shape == (4, 4)
    assert C[0, 0] == 0.0  # Same position
    assert C[0, 1] == 1.0  # Same env, different elem
    assert C[0, 2] == 2.0  # Different env, same elem
    assert C[0, 3] == 3.0  # Different env and elem
    assert C[1, 0] == 1.0
    assert C[1, 3] == 2.0  # Different env, same elem
    assert C[2, 0] == 2.0
    assert C[2, 3] == 1.0  # Same env, different elem


def test_build_cost_matrix_normalized():
    """Test normalized cost matrix."""
    keys = [
        PositionDescriptor(env_id=1, elem="F"),
        PositionDescriptor(env_id=2, elem="Cl"),
    ]

    C = _build_cost_matrix(keys, env_mismatch_cost=2.0, elem_mismatch_cost=1.0, normalize=True)

    assert C.max() <= 1.0
    assert C.min() >= 0.0


def test_build_rbf_kernel():
    """Test RBF kernel construction."""
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    K = _build_rbf_kernel(C, bandwidth=1.0)

    assert K.shape == (2, 2)
    assert K[0, 0] == 1.0  # exp(-0) = 1
    assert K[1, 1] == 1.0
    assert 0.0 < K[0, 1] < 1.0
    assert np.allclose(K[0, 1], K[1, 0])  # Symmetry


def test_build_rbf_kernel_invalid_bandwidth():
    """Test that invalid bandwidth raises error."""
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        _build_rbf_kernel(C, bandwidth=0.0)


def test_sinkhorn_zero_on_identical():
    """Test that Sinkhorn-EMD is near zero for identical distributions."""
    p = np.array([0.3, 0.5, 0.2])
    q = np.array([0.3, 0.5, 0.2])
    C = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ])

    emd = sinkhorn_emd(p, q, C, epsilon=0.1, n_iters=100)
    assert emd < 1e-4, f"Expected near-zero EMD, got {emd}"


def test_sinkhorn_ordering():
    """Test that more different distributions have larger EMD."""
    p = np.array([0.5, 0.3, 0.2])
    q1 = np.array([0.45, 0.35, 0.2])  # Close to p
    q2 = np.array([0.2, 0.3, 0.5])    # Far from p

    C = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ])

    emd1 = sinkhorn_emd(p, q1, C, epsilon=0.1)
    emd2 = sinkhorn_emd(p, q2, C, epsilon=0.1)

    assert emd2 > emd1, f"Expected emd2 ({emd2}) > emd1 ({emd1})"


def test_sinkhorn_invalid_inputs():
    """Test that Sinkhorn raises errors on invalid inputs."""
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5, 0.0])  # Wrong shape
    C = np.zeros((2, 2))

    with pytest.raises(ValueError, match="same shape"):
        sinkhorn_emd(p, q, C)


def test_mmd2_zero_on_identical():
    """Test that MMD2 is near zero for identical distributions."""
    p = np.array([0.3, 0.5, 0.2])
    q = np.array([0.3, 0.5, 0.2])
    K = np.array([
        [1.0, 0.8, 0.5],
        [0.8, 1.0, 0.7],
        [0.5, 0.7, 1.0],
    ])

    mmd2_val = mmd2(p, q, K)
    assert mmd2_val < 1e-8, f"Expected near-zero MMD2, got {mmd2_val}"


def test_mmd2_non_negative():
    """Test that MMD2 is always non-negative."""
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.5, 0.3])
    K = np.array([
        [1.0, 0.8, 0.5],
        [0.8, 1.0, 0.7],
        [0.5, 0.7, 1.0],
    ])

    mmd2_val = mmd2(p, q, K)
    assert mmd2_val >= 0.0


def test_mmd2_invalid_inputs():
    """Test that MMD2 raises errors on invalid inputs."""
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5, 0.0])  # Wrong shape
    K = np.eye(2)

    with pytest.raises(ValueError, match="same shape"):
        mmd2(p, q, K)


def test_compute_macro_metrics_shapes():
    """Test that compute_macro_metrics returns valid metrics."""
    gen_hist = np.array([0.3, 0.5, 0.2])

    # Create a minimal reference
    from xb_align.priors.macro_align import MacroAlignReference

    keys = [
        PositionDescriptor(env_id=1, elem="F"),
        PositionDescriptor(env_id=1, elem="Cl"),
        PositionDescriptor(env_id=2, elem="F"),
    ]
    freqs = np.array([0.4, 0.4, 0.2])
    key_to_index = {k: i for i, k in enumerate(keys)}
    cost_matrix = _build_cost_matrix(keys)
    kernel_matrix = _build_rbf_kernel(cost_matrix)

    ref = MacroAlignReference(
        keys=keys,
        freqs=freqs,
        key_to_index=key_to_index,
        cost_matrix=cost_matrix,
        kernel_matrix=kernel_matrix,
    )

    metrics = compute_macro_metrics(gen_hist, ref)

    assert isinstance(metrics.emd, float)
    assert isinstance(metrics.mmd2, float)
    assert isinstance(metrics.l2, float)
    assert metrics.emd >= 0.0
    assert metrics.mmd2 >= 0.0
    assert metrics.l2 >= 0.0


def test_load_macro_reference_integration(tmp_path):
    """Integration test for loading macro reference from file."""
    # Create a temporary npz file
    keys = np.array([
        PositionDescriptor(env_id=1, elem="F"),
        PositionDescriptor(env_id=1, elem="Cl"),
    ], dtype=object)
    freqs = np.array([0.6, 0.4])

    npz_path = tmp_path / "test_ref.npz"
    np.savez(npz_path, keys=keys, freqs=freqs)

    # Load it
    ref = load_macro_reference(npz_path)

    assert len(ref.keys) == 2
    assert np.allclose(ref.freqs, [0.6, 0.4])
    assert ref.freqs.sum() == pytest.approx(1.0)
    assert ref.cost_matrix.shape == (2, 2)
    assert ref.kernel_matrix.shape == (2, 2)


def test_sinkhorn_simple_2d_ordering():
    """Test EMD ordering with simple 2D synthetic distributions."""
    # Two-bin distributions with simple 0/1 cost matrix
    p_uniform = np.array([0.5, 0.5])
    p_left = np.array([1.0, 0.0])
    p_right = np.array([0.0, 1.0])

    # Cost: moving mass from bin 0 to bin 1 costs 1
    C = np.array([[0.0, 1.0], [1.0, 0.0]])

    # EMD from uniform to uniform should be 0
    emd_uniform = sinkhorn_emd(p_uniform, p_uniform, C, epsilon=0.1)
    assert emd_uniform < 0.01

    # EMD from left to right should be maximal (all mass moves)
    emd_extreme = sinkhorn_emd(p_left, p_right, C, epsilon=0.1)

    # EMD from left to uniform should be intermediate
    emd_intermediate = sinkhorn_emd(p_left, p_uniform, C, epsilon=0.1)

    # Check ordering: extreme > intermediate > uniform
    assert emd_extreme > emd_intermediate, f"Expected {emd_extreme} > {emd_intermediate}"
    assert emd_intermediate > emd_uniform, f"Expected {emd_intermediate} > {emd_uniform}"

    # Approximate expected values (with entropic regularization)
    # left->right: should be close to 1.0 (all mass crosses)
    assert 0.8 < emd_extreme < 1.2, f"EMD left->right should be ~1.0, got {emd_extreme}"
    # left->uniform: should be close to 0.5 (half the mass crosses)
    assert 0.3 < emd_intermediate < 0.7, f"EMD left->uniform should be ~0.5, got {emd_intermediate}"


def test_histogram_consistency_full_vs_changed():
    """Test that histogram_from_mols and histogram_from_changed_atoms give consistent results.

    When changed_atoms includes all allowed atoms, both methods should produce
    similar histograms (allowing for small numerical differences).
    """
    from rdkit import Chem
    from xb_align.priors.macro_align import (
        histogram_from_mols,
        histogram_from_changed_atoms,
        ALLOWED_ELEMS,
    )
    from xb_align.core.env_featurizer import SimpleEnvFeaturizer

    # Create test molecules with known elements
    test_smiles = [
        "c1c(F)c(Cl)c(N)cc1",  # Fluorine, Chlorine, Nitrogen
        "c1cc(O)c(Br)cc1",      # Oxygen, Bromine
    ]

    mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]

    # Create a simple reference
    keys = [
        PositionDescriptor(env_id=1, elem="F"),
        PositionDescriptor(env_id=2, elem="Cl"),
        PositionDescriptor(env_id=3, elem="N"),
        PositionDescriptor(env_id=4, elem="O"),
        PositionDescriptor(env_id=5, elem="Br"),
    ]
    freqs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    key_to_index = {k: i for i, k in enumerate(keys)}
    cost_matrix = _build_cost_matrix(keys)
    kernel_matrix = _build_rbf_kernel(cost_matrix)

    ref = MacroAlignReference(
        keys=keys,
        freqs=freqs,
        key_to_index=key_to_index,
        cost_matrix=cost_matrix,
        kernel_matrix=kernel_matrix,
    )

    env_featurizer = SimpleEnvFeaturizer()

    # Method 1: histogram_from_mols (all allowed atoms)
    hist_full = histogram_from_mols(mols, env_featurizer, ref, ALLOWED_ELEMS)

    # Method 2: histogram_from_changed_atoms with all allowed atoms marked
    samples = []
    for mol in mols:
        if mol is None:
            continue
        # Find all atoms with allowed elements
        changed_atoms = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() in ALLOWED_ELEMS
        ]
        samples.append((mol, changed_atoms))

    hist_changed = histogram_from_changed_atoms(samples, env_featurizer, ref)

    # Both histograms should be identical (or very close)
    # They count the same atoms, just with different interfaces
    if hist_full.sum() > 0 and hist_changed.sum() > 0:
        # Allow small numerical differences due to normalization
        assert np.allclose(hist_full, hist_changed, atol=1e-10), \
            f"Histograms should match:\nFull: {hist_full}\nChanged: {hist_changed}"


def test_load_macro_reference_real_data():
    """Test loading real drug_halopos_ref.npz if it exists."""
    ref_path = Path("data/processed/drug_halopos_ref.npz")
    if not ref_path.exists():
        pytest.skip("drug_halopos_ref.npz not found")

    ref = load_macro_reference(ref_path)

    # Verify structure
    assert len(ref.keys) > 0
    assert ref.freqs.shape[0] == len(ref.keys)
    assert ref.freqs.sum() == pytest.approx(1.0)
    assert ref.cost_matrix.shape[0] == len(ref.keys)
    assert ref.kernel_matrix.shape[0] == len(ref.keys)

    # Test that identical distribution has near-zero metrics
    # Note: Sinkhorn with entropic regularization has some numerical error
    metrics = compute_macro_metrics(ref.freqs, ref, epsilon=0.1)
    assert metrics.emd < 0.01, f"EMD should be small for identical distributions, got {metrics.emd}"
    assert metrics.mmd2 < 1e-8
    assert metrics.l2 < 1e-12


def test_union_support_all_oov():
    """Test that union support produces non-zero EMD even when all baseline keys are OOV.

    This tests the fix for the EMD=0 bug where baseline molecules had 100% OOV rate,
    causing the histogram to be all zeros and EMD to collapse to 0.
    """
    # Create reference with 3 position descriptors
    ref_keys = [
        PositionDescriptor(env_id=100, elem="F"),
        PositionDescriptor(env_id=101, elem="Cl"),
        PositionDescriptor(env_id=102, elem="Br"),
    ]
    ref_freqs = np.array([0.5, 0.3, 0.2])
    key_to_index = {k: i for i, k in enumerate(ref_keys)}
    cost_matrix = _build_cost_matrix(ref_keys)
    kernel_matrix = _build_rbf_kernel(cost_matrix)

    ref = MacroAlignReference(
        keys=ref_keys,
        freqs=ref_freqs,
        key_to_index=key_to_index,
        cost_matrix=cost_matrix,
        kernel_matrix=kernel_matrix,
    )

    # Create baseline counts with COMPLETELY DIFFERENT keys (100% OOV)
    baseline_counts = {
        PositionDescriptor(env_id=200, elem="N"): 10,
        PositionDescriptor(env_id=201, elem="O"): 15,
        PositionDescriptor(env_id=202, elem="S"): 5,
    }

    # Test build_union_support
    p_all, q_all, C_all, K_all = build_union_support(
        ref=ref,
        baseline_counts=baseline_counts,
        debug=False,
    )

    # Verify union support has 6 keys (3 ref + 3 baseline)
    assert len(p_all) == 6
    assert len(q_all) == 6
    assert C_all.shape == (6, 6)
    assert K_all.shape == (6, 6)

    # Verify p_all (baseline) has mass only in last 3 positions
    assert np.allclose(p_all[:3], 0.0)
    assert p_all[3:].sum() == pytest.approx(1.0)

    # Verify q_all (ref) has mass only in first 3 positions
    assert q_all[:3].sum() == pytest.approx(1.0)
    assert np.allclose(q_all[3:], 0.0)

    # Compute metrics on union support
    metrics = compute_macro_metrics_union(
        baseline_counts=baseline_counts,
        ref=ref,
        debug=False,
    )

    # CRITICAL: EMD should be NON-ZERO because distributions are different
    # Even though vocabularies don't overlap, transport cost is non-zero
    assert metrics.emd > 0.0, f"EMD should be > 0 for non-overlapping distributions, got {metrics.emd}"

    # MMD and L2 should also be non-zero
    assert metrics.mmd2 > 0.0
    assert metrics.l2 > 0.0

    # Verify EMD is substantial (distributions are completely different)
    # With complete OOV and different elements, EMD should be relatively large
    assert metrics.emd > 0.5, f"Expected substantial EMD for completely different distributions, got {metrics.emd}"


def test_union_support_partial_overlap():
    """Test union support with partial vocabulary overlap.

    This tests the more realistic case where baseline and reference share some keys
    but baseline also has OOV keys.
    """
    # Create reference with 4 position descriptors
    ref_keys = [
        PositionDescriptor(env_id=100, elem="F"),
        PositionDescriptor(env_id=101, elem="Cl"),
        PositionDescriptor(env_id=102, elem="Br"),
        PositionDescriptor(env_id=103, elem="I"),
    ]
    ref_freqs = np.array([0.4, 0.3, 0.2, 0.1])
    key_to_index = {k: i for i, k in enumerate(ref_keys)}
    cost_matrix = _build_cost_matrix(ref_keys)
    kernel_matrix = _build_rbf_kernel(cost_matrix)

    ref = MacroAlignReference(
        keys=ref_keys,
        freqs=ref_freqs,
        key_to_index=key_to_index,
        cost_matrix=cost_matrix,
        kernel_matrix=kernel_matrix,
    )

    # Create baseline counts with 2 shared keys + 2 new keys (50% overlap)
    baseline_counts = {
        PositionDescriptor(env_id=100, elem="F"): 20,    # Shared
        PositionDescriptor(env_id=101, elem="Cl"): 15,   # Shared
        PositionDescriptor(env_id=200, elem="N"): 10,    # OOV
        PositionDescriptor(env_id=201, elem="O"): 5,     # OOV
    }

    # Test build_union_support
    p_all, q_all, C_all, K_all = build_union_support(
        ref=ref,
        baseline_counts=baseline_counts,
        debug=False,
    )

    # Verify union support has 6 keys (4 ref + 2 baseline-only)
    assert len(p_all) == 6
    assert len(q_all) == 6

    # Verify baseline has mass in positions 0, 1 (shared) and 4, 5 (OOV)
    total_baseline_mass = sum(baseline_counts.values())
    expected_p0 = 20 / total_baseline_mass
    expected_p1 = 15 / total_baseline_mass
    expected_p4 = 10 / total_baseline_mass
    expected_p5 = 5 / total_baseline_mass

    assert p_all[0] == pytest.approx(expected_p0)
    assert p_all[1] == pytest.approx(expected_p1)
    assert p_all[4] == pytest.approx(expected_p4)
    assert p_all[5] == pytest.approx(expected_p5)
    assert p_all.sum() == pytest.approx(1.0)

    # Verify ref has mass only in first 4 positions
    assert np.allclose(q_all[:4], ref_freqs)
    assert np.allclose(q_all[4:], 0.0)

    # Compute metrics
    metrics = compute_macro_metrics_union(
        baseline_counts=baseline_counts,
        ref=ref,
        debug=False,
    )

    # EMD should be non-zero (distributions differ)
    assert metrics.emd > 0.0

    # Compare with a perfectly matching distribution
    # If baseline matched ref exactly, EMD should be smaller
    perfect_match_counts = {
        PositionDescriptor(env_id=100, elem="F"): 40,
        PositionDescriptor(env_id=101, elem="Cl"): 30,
        PositionDescriptor(env_id=102, elem="Br"): 20,
        PositionDescriptor(env_id=103, elem="I"): 10,
    }

    metrics_perfect = compute_macro_metrics_union(
        baseline_counts=perfect_match_counts,
        ref=ref,
        debug=False,
    )

    # Perfect match should have lower EMD than partial overlap
    assert metrics_perfect.emd < metrics.emd, \
        f"Perfect match EMD ({metrics_perfect.emd}) should be < partial overlap EMD ({metrics.emd})"

    # Perfect match should have near-zero EMD
    assert metrics_perfect.emd < 0.01, \
        f"Perfect match should have near-zero EMD, got {metrics_perfect.emd}"
