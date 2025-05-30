import numpy as np
from collections import deque
import torch

class PreProcessor():
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.grayscale_weights = torch.tensor(
            [0.299, 0.587, 0.114], 
            dtype=torch.float32, 
            device=self.device
        )


    def process(self, states_deque: deque) -> torch.Tensor:
        grayscaled_states = self._convert_to_grayscale(states_deque)
        return grayscaled_states.to(device=self.device)


    def _convert_to_grayscale(self, states_deque: deque) -> torch.Tensor:
        states_array = np.array(list(states_deque))
        
        states_tensor = torch.from_numpy(states_array).float()

        # Holy shit, das Ding ist crazy!
        # einsum ist die Einsteinsumme, damit lassen sich Tensoren beliebiger Form
        # miteinander multiplizieren und aufsummieren
        # Die Formel für eine Einsteinsumme besteht aus dem linken Teil (links vom ->),
        # der durch ein Komma getrennt die shapes der Tensoren angibt.
        # n ist Anzahl an Batches, h ist Höhe, w Beite und c sind die Channel
        # Gleiche Buchstaben werden miteinander multipliziert (in diesem Fall also c und c)
        # Die rechte Seite gibt an, welche Dimensionen übrig bleiben.
        grayscale_states = torch.einsum(
            "nhwc,c->nhw",
            states_tensor[..., :3], 
            self.grayscale_weights.to(states_tensor.device)).byte()
        
        return grayscale_states