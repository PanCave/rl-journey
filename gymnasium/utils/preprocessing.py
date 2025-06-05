from collections import deque

from numpy.typing import NDArray
import torch
import torchvision.transforms.functional  as F


def convert_to_grayscale(state: NDArray) -> torch.Tensor:
    # input: whc -> output: chw
    state_tensor = torch.tensor(state).permute(2, 1, 0).unsqueeze(0)
    grayscaled_state = F.rgb_to_grayscale(state_tensor)

    return grayscaled_state.squeeze()

def deque_to_tensor(queue: deque) -> torch.Tensor:
    return torch.stack(list(queue))