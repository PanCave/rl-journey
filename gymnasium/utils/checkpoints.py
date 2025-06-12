import os

from pathlib import Path
from typing import Any

import torch

from agents.discrete_agent import DiscreteAgent

def save_checkpoint(
        agent: DiscreteAgent,
        episode_idx: int,
        save_checkpoint_path_str: str) -> None:
    
    path = Path(save_checkpoint_path_str)
    directory = path.parents[0]
    directory.mkdir(exist_ok=True, parents=True)
    
    save_dict = {
        'policy_network_state_dict': agent.policy_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'epsilon_init': agent.epsilon_init,
        'episode_idx': episode_idx
    }
    torch.save(save_dict, save_checkpoint_path_str)

def load_checkpoint(load_checkpoint_path: str) -> dict[str: Any] | None:
    if os.path.exists(load_checkpoint_path):
        print("Checkpoint found!")
        return torch.load(load_checkpoint_path, weights_only=False)
    
    return None