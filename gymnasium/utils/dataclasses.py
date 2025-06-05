from dataclasses import dataclass

import torch

@dataclass
class Replay:
    state : torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool