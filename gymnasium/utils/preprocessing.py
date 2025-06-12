from collections import deque
from typing import Tuple

from numpy.typing import NDArray
import torch
import torchvision.transforms.functional  as F


def are_states_equal(
    first_state: torch.Tensor,
    second_state: torch.Tensor,
    x_min: int, x_max: int,
    y_min: int, y_max: int
    ) -> bool:
    height, width = first_state.shape[-2], first_state.shape[-1]
    
    y_indices, x_indices = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij"
    )
    
    ignore_x = (x_indices >= x_min) & (x_indices <= x_max)
    ignore_y = (y_indices >= y_min) & (y_indices <= y_max)
    ignore_mask = ignore_x & ignore_y    
    compare_mask = ~ignore_mask
    
    masked_first_state = first_state[compare_mask]
    maskes_second_state = second_state[compare_mask]
    
    return torch.allclose(masked_first_state, maskes_second_state, rtol=1e-5, atol=1e-8)

def convert_to_grayscale(state: NDArray, slices: Tuple[slice, ...]) -> torch.Tensor:
    # input: whc -> output: chw
    state = state[slices]
    state_tensor = torch.tensor(state).permute(2, 1, 0).unsqueeze(0)
    grayscaled_state = F.rgb_to_grayscale(state_tensor)

    return grayscaled_state.squeeze()

def convert_to_tensor(state: NDArray, device: torch.device) -> torch.Tensor:
    return torch.tensor(state, device=device)

def deque_to_tensor(queue: deque) -> torch.Tensor:
    return torch.stack(list(queue))