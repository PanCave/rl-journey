import os

from pathlib import Path
from typing import Any

import torch

from agents.discrete_agent import DiscreteAgent
from agents.continuous_agent import SACAgent

def save_sac_checkpoint(
    agent: SACAgent,
    episode_idx: int,
    save_checkpoint_path_str: str
) -> None:
    path = Path(save_checkpoint_path_str)
    directory = path.parents[0]
    directory.mkdir(exist_ok=True, parents=True)
    
    save_dict = {
        'policy_network_state_dict': agent.policy_network.state_dict(),
        'critic_1_network_state_dict': agent.critic_1_network.state_dict(),
        'critic_2_network_state_dict': agent.critic_2_network.state_dict(),
        'target_1_network_state_dict': agent.target_1_network.state_dict(),
        'target_2_network_state_dict': agent.target_2_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }
    torch.save(save_dict, save_checkpoint_path_str)

def save_dqn_checkpoint(
        agent: DiscreteAgent,
        episode_idx: int,
        save_checkpoint_path_str: str) -> None:
    
    path = Path(save_checkpoint_path_str)
    directory = path.parents[0]
    directory.mkdir(exist_ok=True, parents=True)
    
    save_dict = {
        'policy_network_state_dict': agent.policy_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }
    torch.save(save_dict, save_checkpoint_path_str)

def load_checkpoint(load_checkpoint_path: str) -> dict[str, Any] | None:
    if os.path.exists(load_checkpoint_path):
        print("Checkpoint found!")
        return torch.load(load_checkpoint_path, weights_only=False)
    
    return None