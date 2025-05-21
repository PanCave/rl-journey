from dataclasses import dataclass

from numpy.typing import NDArray

@dataclass
class Replay:
    state : NDArray
    action: int
    reward: float
    next_state: NDArray