# tests/test_macro_eval.py
"""Tests for macro evaluation module."""

import pytest
import numpy as np
from pathlib import Path
from rdkit import Chem

from xb_align.eval.macro_eval import (
    compute_baseline_macro_hist,
    plot_macro_hist_comparison,
)
from xb_align.baseline.generator import BaselineSample
from xb_align.priors.macro_align import (
    MacroAlignReference,
    _build_cost_matrix,
    _build_rbf_kernel,
)
from xb_align.priors.position_descriptor import PositionDescriptor
from xb_align.core.env_featurizer import SimpleEnvFeaturizer


@pytest.fixture
def mock_reference():
    """Create a minimal mock reference for testing."""
    keys = [
        PositionDescriptor(env_id=1, elem="F"),
        PositionDescriptor(env_id=1, elem="Cl"),
        PositionDescriptor(env_id=2, elem="N"),
    ]
    freqs = np.array([0.4, 0.3, 0.3])
    key_to_index = {k: i for i, k in enumerate(keys)}
    cost_matrix = _build_cost_matrix(keys)
    kernel_matrix = _build_rbf_kernel(cost_matrix)

    return MacroAlignReference(
        keys=keys,
        freqs=freqs,
        key_to_index=key_to_index,
        cost_matrix=cost_matrix,
        kernel_matrix=kernel_matrix,
    )


def test_compute_baseline_macro_hist_empty():
    """Test histogram computation with empty samples."""
    samples = []
    env_featurizer = SimpleEnvFeaturizer()

    keys = [PositionDescriptor(env_id=1, elem="F")]
    ref = MacroAlignReference(
        keys=keys,
        freqs=np.array([1.0]),
        key_to_index={keys[0]: 0},
        cost_matrix=np.zeros((1, 1)),
        kernel_matrix=np.ones((1, 1)),
    )

    hist = compute_baseline_macro_hist(samples, ref, env_featurizer)

    assert hist.shape == (1,)
    assert hist.sum() == 0.0  # No samples, all zeros


def test_compute_baseline_macro_hist_simple(mock_reference):
    """Test histogram computation with simple samples."""
    # Create samples with known changed atoms
    samples = [
        BaselineSample(0, "c1ccccc1", "c1c(F)cccc1", [1]),
        BaselineSample(1, "c1ccccc1", "c1cc(F)ccc1", [2]),
    ]

    env_featurizer = SimpleEnvFeaturizer()
    hist = compute_baseline_macro_hist(samples, mock_reference, env_featurizer)

    # Histogram should sum to 1 (normalized)
    assert hist.shape == (len(mock_reference.keys),)
    if hist.sum() > 0:
        assert hist.sum() == pytest.approx(1.0)


def test_compute_baseline_macro_hist_invalid_smiles(mock_reference):
    """Test that invalid SMILES are skipped."""
    samples = [
        BaselineSample(0, "c1ccccc1", "INVALID", [1]),
        BaselineSample(1, "c1ccccc1", "c1c(F)cccc1", [1]),
    ]

    env_featurizer = SimpleEnvFeaturizer()
    hist = compute_baseline_macro_hist(samples, mock_reference, env_featurizer)

    # Should still return a valid histogram
    assert hist.shape == (len(mock_reference.keys),)


def test_plot_macro_hist_comparison_creates_file(tmp_path, mock_reference):
    """Test that plotting creates an output file."""
    gen_hist = np.array([0.3, 0.4, 0.3])
    out_path = tmp_path / "test_plot.png"

    plot_macro_hist_comparison(
        ref=mock_reference,
        gen_hist=gen_hist,
        out_path=out_path,
        top_k=3,
    )

    assert out_path.exists()


def test_plot_macro_hist_comparison_invalid_shape(mock_reference):
    """Test that mismatched histogram shape raises error."""
    gen_hist = np.array([0.5, 0.5])  # Wrong shape
    out_path = Path("dummy.png")

    with pytest.raises(ValueError, match="does not match reference"):
        plot_macro_hist_comparison(
            ref=mock_reference,
            gen_hist=gen_hist,
            out_path=out_path,
        )


def test_plot_macro_hist_comparison_top_k(tmp_path, mock_reference):
    """Test that top_k parameter limits the plot."""
    gen_hist = np.array([0.3, 0.4, 0.3])
    out_path = tmp_path / "test_plot_top1.png"

    # Should not raise error even if top_k < len(keys)
    plot_macro_hist_comparison(
        ref=mock_reference,
        gen_hist=gen_hist,
        out_path=out_path,
        top_k=1,
    )

    assert out_path.exists()


def test_plot_macro_hist_creates_parent_dirs(tmp_path, mock_reference):
    """Test that plot function creates parent directories."""
    gen_hist = np.array([0.3, 0.4, 0.3])
    out_path = tmp_path / "subdir" / "nested" / "plot.png"

    plot_macro_hist_comparison(
        ref=mock_reference,
        gen_hist=gen_hist,
        out_path=out_path,
    )

    assert out_path.exists()
    assert out_path.parent.exists()
