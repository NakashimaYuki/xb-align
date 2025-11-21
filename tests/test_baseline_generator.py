# tests/test_baseline_generator.py
"""Tests for baseline generator."""

import pytest
import pandas as pd
from pathlib import Path
from rdkit import Chem

from xb_align.baseline.generator import (
    BaselineSample,
    load_np_scaffolds,
    sample_baseline,
)


def test_baseline_sample_dataclass():
    """Test BaselineSample dataclass."""
    sample = BaselineSample(
        scaffold_id=0,
        scaffold_smiles="c1ccccc1",
        generated_smiles="c1ccc(N)cc1",
        changed_atoms=[3],
    )

    assert sample.scaffold_id == 0
    assert sample.scaffold_smiles == "c1ccccc1"
    assert sample.generated_smiles == "c1ccc(N)cc1"
    assert sample.changed_atoms == [3]


def test_load_np_scaffolds_real():
    """Test loading real np_scaffolds.parquet if it exists."""
    path = Path("data/processed/np_scaffolds.parquet")
    if not path.exists():
        pytest.skip("np_scaffolds.parquet not found")

    df = load_np_scaffolds(path)

    assert isinstance(df, pd.DataFrame)
    assert "scaffold_smiles" in df.columns
    assert len(df) > 0


def test_sample_baseline_simple():
    """Test baseline sampling with a small fake scaffold library."""
    # Create a simple fake scaffold library
    scaffolds = pd.DataFrame({
        "scaffold_smiles": ["c1ccccc1", "CCO", "c1ccc2ccccc2c1"],
    })

    samples = sample_baseline(
        scaffolds=scaffolds,
        n_samples=10,
        max_changes=2,
        seed=42,
    )

    # Should generate some samples
    assert len(samples) > 0
    assert all(isinstance(s, BaselineSample) for s in samples)


def test_sample_baseline_deterministic():
    """Test that sampling is deterministic with fixed seed."""
    scaffolds = pd.DataFrame({
        "scaffold_smiles": ["c1ccccc1", "CCO", "c1ccc2ccccc2c1"],
    })

    samples1 = sample_baseline(scaffolds, n_samples=5, max_changes=2, seed=123)
    samples2 = sample_baseline(scaffolds, n_samples=5, max_changes=2, seed=123)

    # Same seed should give same results
    assert len(samples1) == len(samples2)
    for s1, s2 in zip(samples1, samples2):
        assert s1.scaffold_id == s2.scaffold_id
        assert s1.scaffold_smiles == s2.scaffold_smiles
        assert s1.generated_smiles == s2.generated_smiles
        assert s1.changed_atoms == s2.changed_atoms


def test_sample_baseline_valid_molecules():
    """Test that all generated molecules are chemically valid."""
    scaffolds = pd.DataFrame({
        "scaffold_smiles": ["c1ccccc1", "CCO", "c1ccc2ccccc2c1"],
    })

    samples = sample_baseline(scaffolds, n_samples=20, max_changes=3, seed=42)

    for sample in samples:
        mol = Chem.MolFromSmiles(sample.generated_smiles)
        assert mol is not None, f"Invalid SMILES: {sample.generated_smiles}"

        # Should pass sanitization
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            pytest.fail(f"Sanitization failed for {sample.generated_smiles}: {e}")


def test_sample_baseline_changed_atoms_valid():
    """Test that changed_atoms are within valid range."""
    scaffolds = pd.DataFrame({
        "scaffold_smiles": ["c1ccccc1", "CCO"],
    })

    samples = sample_baseline(scaffolds, n_samples=10, max_changes=2, seed=42)

    for sample in samples:
        mol = Chem.MolFromSmiles(sample.generated_smiles)
        assert mol is not None

        num_atoms = mol.GetNumAtoms()
        for idx in sample.changed_atoms:
            assert 0 <= idx < num_atoms, f"Invalid atom index {idx} for molecule with {num_atoms} atoms"


def test_sample_baseline_missing_column():
    """Test that missing scaffold_smiles column raises error."""
    scaffolds = pd.DataFrame({
        "smiles": ["c1ccccc1"],  # Wrong column name
    })

    with pytest.raises(ValueError, match="scaffold_smiles"):
        sample_baseline(scaffolds, n_samples=5, max_changes=2)


def test_sample_baseline_respects_n_samples():
    """Test that at most n_samples are generated."""
    scaffolds = pd.DataFrame({
        "scaffold_smiles": ["c1ccccc1", "CCO", "c1ccc2ccccc2c1"],
    })

    n_samples = 15
    samples = sample_baseline(scaffolds, n_samples=n_samples, max_changes=3, seed=42)

    # Should not exceed requested number
    assert len(samples) <= n_samples


def test_sample_baseline_scaffold_id_valid():
    """Test that scaffold_id references valid scaffold indices."""
    scaffolds = pd.DataFrame({
        "scaffold_smiles": ["c1ccccc1", "CCO", "c1ccc2ccccc2c1"],
    })

    samples = sample_baseline(scaffolds, n_samples=10, max_changes=2, seed=42)

    valid_ids = set(scaffolds.index)
    for sample in samples:
        assert sample.scaffold_id in valid_ids
        # Verify scaffold_smiles matches the ID
        assert sample.scaffold_smiles == scaffolds.loc[sample.scaffold_id, "scaffold_smiles"]
