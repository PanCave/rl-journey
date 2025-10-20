from typing import List
import torch.nn as nn
import gymnasium as gym
import torch
import numpy as np
from copy import deepcopy

from utils.dataclasses import Replay

class SACAgent:
    def __init__(
        self,
        env: gym.Env,
        num_target_update_steps: int,
        policy_network: nn.Module,
        critic_1_network: nn.Module,
        critic_2_network: nn.Module,
        action_dim: int,
        alpha: float,
        tau: float,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> None:
        self.env = env
        self.num_target_update_steps = num_target_update_steps
        self.policy_network = policy_network
        self.critic_1_network = critic_1_network
        self.target_1_network = deepcopy(critic_1_network)
        self.critic_2_network = critic_2_network
        self.target_2_network = deepcopy(critic_2_network)
        self.device = device
        self.target_net_update_step_counter = 0
        
        self.action_dim = action_dim
        self.alpha = alpha
        self.tau = tau

        self.optimizer = optimizer
        self.policy_network.to(device=self.device)
        #self.target_network.to(device=self.device)
        
    def select_action(
        self,
        state: torch.Tensor,
        inference_only: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            self.policy_network.eval()
            state = torch.unsqueeze(state, 0).to(device=self.device)
            mu_sigma_values: torch.Tensor = self.policy_network.forward(state)
            mu_sigma_values = mu_sigma_values.squeeze()
            mu = mu_sigma_values.numpy()[::2]
            
        if inference_only:
            return mu.tolist()
        else:
            sigma = mu_sigma_values.numpy()[1::2]
            #action_dim_tensor = torch.tensor(self.action_dim, device=self.device)
            e = np.random.normal(self.action_dim)
            u = mu + sigma * e
            a = np.tanh(u)
            
            return a
        
    
    def train(self, replay_batch: List[Replay]) -> int:
        return 0