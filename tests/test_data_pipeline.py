# tests/test_data_pipeline.py
"""Tests for data processing pipeline."""

import pytest
from rdkit import Chem

from xb_align.data.prepare_np_scaffolds import normalize_smiles, get_scaffold_smiles, calc_scaffold_props
from xb_align.core.env_featurizer import SimpleEnvFeaturizer
from xb_align.priors.position_descriptor import PositionDescriptor


def test_normalize_smiles():
    """Test SMILES normalization."""
    smi = "Clc1ccccc1"
    norm = normalize_smiles(smi)
    assert isinstance(norm, str)
    m = Chem.MolFromSmiles(norm)
    assert m is not None
    assert m.GetNumAtoms() > 0


def test_normalize_smiles_invalid():
    """Test that invalid SMILES returns None."""
    assert normalize_smiles("invalid_smiles_xyz") is None
    assert normalize_smiles(None) is None
    assert normalize_smiles(123) is None


def test_get_scaffold_smiles():
    """Test scaffold extraction."""
    smi = "CCOc1ccccc1Cl"
    scaf = get_scaffold_smiles(smi)
    assert isinstance(scaf, str)
    m = Chem.MolFromSmiles(scaf)
    assert m is not None
    # Scaffold should have fewer atoms than original
    orig = Chem.MolFromSmiles(smi)
    assert m.GetNumAtoms() <= orig.GetNumAtoms()


def test_calc_scaffold_props():
    """Test scaffold property calculation."""
    scaf_smi = "c1ccccc1"
    props = calc_scaffold_props(scaf_smi)
    assert "fsp3" in props
    assert "n_rings" in props
    assert "n_atoms" in props
    assert props["n_rings"] == 1  # benzene has 1 ring
    assert props["n_atoms"] == 6  # benzene has 6 atoms


def test_env_featurizer():
    """Test environment featurizer."""
    mol = Chem.MolFromSmiles("c1ccccc1Cl")
    featurizer = SimpleEnvFeaturizer()
    env_id = featurizer.encode(mol, 0)
    assert isinstance(env_id, int)
    assert env_id >= 0


def test_position_descriptor():
    """Test PositionDescriptor dataclass."""
    desc = PositionDescriptor(env_id=12345, elem="Cl")
    assert desc.env_id == 12345
    assert desc.elem == "Cl"
    # Test immutability
    with pytest.raises(Exception):
        desc.env_id = 99999
