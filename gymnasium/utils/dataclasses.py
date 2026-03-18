from dataclasses import dataclass

from numpy.typing import NDArray

import torch

@dataclass
class Replay:
    state : torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    terminated: bool

@dataclass
class ReplayContinuous:
    state: torch.Tensor
    action: NDArray
    reward: float
    next_state: torch.Tensor
    terminated: bool