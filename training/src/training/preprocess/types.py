### Dataclass for Intermediate Trajectory Representation
from dataclasses import dataclass
from typing import List


@dataclass
class ParsedTrajectory:
    """Intermediate representation of a game trajectory before tensor conversion."""
    states: List[List[int]]  # List of 4x4 states with log2 applied
    actions: List[int]        # List of action indices (0-3)
    length: int               # Number of steps in the trajectory