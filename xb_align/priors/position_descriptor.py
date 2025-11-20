# xb_align/priors/position_descriptor.py
"""Position descriptor for halogen/heteroatom locations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionDescriptor:
    """Descriptor for a specific atom position in a molecule.

    Attributes:
        env_id: Environment ID encoding local structure
        elem: Element symbol (e.g., 'F', 'Cl', 'N', 'O')
    """
    env_id: int
    elem: str
