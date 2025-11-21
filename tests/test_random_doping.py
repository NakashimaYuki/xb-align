# tests/test_random_doping.py
"""Tests for random doping baseline."""

import random
import pytest
from rdkit import Chem

from xb_align.baseline.random_doping import random_single_atom_doping, ALLOWED_SUB_ELEMS


def test_random_doping_simple_scaffold():
    """Test random doping on benzene."""
    scaffold_smi = "c1ccccc1"
    rng = random.Random(42)

    mol, changed_atoms = random_single_atom_doping(
        scaffold_smiles=scaffold_smi,
        max_changes=3,
        rng=rng,
    )

    # Should successfully generate a molecule
    assert mol is not None
    assert len(changed_atoms) > 0
    assert len(changed_atoms) <= 3

    # Verify molecule is valid
    Chem.SanitizeMol(mol)
    smi = Chem.MolToSmiles(mol)
    assert smi is not None


def test_random_doping_deterministic():
    """Test that random doping is deterministic with fixed seed."""
    scaffold_smi = "c1ccccc1"

    rng1 = random.Random(123)
    mol1, changed1 = random_single_atom_doping(scaffold_smi, max_changes=2, rng=rng1)

    rng2 = random.Random(123)
    mol2, changed2 = random_single_atom_doping(scaffold_smi, max_changes=2, rng=rng2)

    # Same seed should give same result
    if mol1 is not None and mol2 is not None:
        smi1 = Chem.MolToSmiles(mol1)
        smi2 = Chem.MolToSmiles(mol2)
        assert smi1 == smi2
        assert changed1 == changed2


def test_random_doping_changed_atoms_valid():
    """Test that changed atoms are within valid range."""
    scaffold_smi = "CCO"
    rng = random.Random(42)

    mol, changed_atoms = random_single_atom_doping(scaffold_smi, max_changes=2, rng=rng)

    if mol is not None:
        num_atoms = mol.GetNumAtoms()
        for idx in changed_atoms:
            assert 0 <= idx < num_atoms


def test_random_doping_elements_in_allowed():
    """Test that substituted elements are from allowed set."""
    scaffold_smi = "c1ccccc1"
    rng = random.Random(42)

    mol, changed_atoms = random_single_atom_doping(scaffold_smi, max_changes=3, rng=rng)

    if mol is not None and changed_atoms:
        for idx in changed_atoms:
            atom = mol.GetAtomWithIdx(idx)
            elem = atom.GetSymbol()
            # Element should be in allowed set
            assert elem in ALLOWED_SUB_ELEMS


def test_random_doping_invalid_smiles():
    """Test handling of invalid SMILES."""
    mol, changed_atoms = random_single_atom_doping("INVALID_SMILES", max_changes=2)

    assert mol is None
    assert changed_atoms == []


def test_random_doping_multiple_runs():
    """Test that multiple runs can produce different results."""
    scaffold_smi = "c1ccccc1"
    results = []

    for seed in range(10):
        rng = random.Random(seed)
        mol, changed_atoms = random_single_atom_doping(scaffold_smi, max_changes=3, rng=rng)
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            results.append(smi)

    # At least some diversity in results
    unique_results = set(results)
    assert len(unique_results) > 1, "Expected some diversity in random doping results"


def test_random_doping_respects_max_changes():
    """Test that number of changes does not exceed max_changes."""
    scaffold_smi = "c1ccc2ccccc2c1"  # Naphthalene (more atoms)
    rng = random.Random(42)

    max_changes = 2
    mol, changed_atoms = random_single_atom_doping(scaffold_smi, max_changes=max_changes, rng=rng)

    if mol is not None:
        assert len(changed_atoms) <= max_changes


def test_random_doping_sanitize_check():
    """Test that returned molecules are chemically valid."""
    scaffold_smi = "CCO"
    rng = random.Random(42)

    # Try multiple times to ensure consistent validity
    for _ in range(5):
        mol, changed_atoms = random_single_atom_doping(scaffold_smi, max_changes=2, rng=rng)
        if mol is not None:
            # Should not raise exception
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                pytest.fail(f"Generated molecule failed sanitization: {e}")
