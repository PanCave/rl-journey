from numpy.typing import NDArray
import torch
import torchvision.transforms.functional  as F

class Preprocessor():
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def convert_to_grayscale(self, state: NDArray) -> torch.Tensor:
        # input: whc -> output: chw
        state_tensor = torch.tensor(state, device=self.device).permute(2, 1, 0).unsqueeze(0)
        grayscaled_state = F.rgb_to_grayscale(state_tensor)

        return grayscaled_state.squeeze()