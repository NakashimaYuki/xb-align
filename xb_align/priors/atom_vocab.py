# xb_align/priors/atom_vocab.py
"""Atom type vocabulary for Graph-MLM."""

ATOM_TYPES = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]

ATOM2IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
IDX2ATOM = {i: a for a, i in ATOM2IDX.items()}
MASK_TOKEN_IDX = len(ATOM_TYPES)
NUM_ATOM_CLASSES = len(ATOM_TYPES) + 1  # include [MASK] token
