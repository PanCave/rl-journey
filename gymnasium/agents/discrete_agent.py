import random

import gymnasium as gym
import numpy as np
from copy import deepcopy
from typing import List
import torch.nn.functional as F
import torch.nn as nn
import torch

from utils.dataclasses import Replay

class DiscreteAgent:
    def __init__(
        self,
        env: gym.Env,
        num_target_update_steps: int,
        policy_network: nn.Module,
        epsilon_init: float,
        epsilon_min: float,
        epsilon_decay_rate: float,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: float
    ) -> None:
        self.env = env
        self.num_target_update_steps = num_target_update_steps
        self.policy_network = policy_network
        self.target_network = deepcopy(policy_network)
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_init
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.device = device
        self.target_net_update_step_counter = 0

        self.target_network.eval()
        self.optimizer = optimizer
        self.policy_network.to(device=self.device)
        self.target_network.to(device=self.device)

    def select_action(
        self,
        state: torch.Tensor,
        inference_only: bool = False
    ) -> int:
        value = random.random()
        if inference_only or value > self.epsilon:
            with torch.no_grad():
                self.policy_network.eval()
                state = torch.unsqueeze(state, 0).to(device=self.device)
                q_values = self.policy_network.forward(state)
                action = int(torch.argmax(q_values))
                return action
        else:
            action = self.env.action_space.sample()
            return action

    def reset_epsilon(self):
        self.epsilon = self.epsilon_init

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def train(self,
              replay_batch: List[Replay]) -> int:
        # Update target network after n steps
        self.target_net_update_step_counter += 1
        if (self.target_net_update_step_counter == self.num_target_update_steps):
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.target_net_update_step_counter = 0

        self.policy_network.train()

        # Get q_values
        states = np.array([replay.state for replay in replay_batch])
        states_tensor = torch.tensor(states, device=self.device)
        actions = np.array([replay.action for replay in replay_batch])
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        q_values_batch = self.policy_network.forward(states_tensor)
        indexes = torch.arange(q_values_batch.size(0), device=self.device)
        q_values = q_values_batch[indexes, actions_tensor]

        # Get q*_values
        next_states = np.array([replay.next_state for replay in replay_batch])
        next_states_tensor = torch.tensor(next_states, device=self.device)

        with torch.no_grad():
            terminated_mask = np.array([replay.terminated for replay in replay_batch])
            terminated_mask_tensor = torch.tensor(terminated_mask, device=self.device, dtype=torch.bool)
            next_q_values = self.target_network.forward(next_states_tensor)

        max_next_q_values = torch.max(
            input = next_q_values,
            dim = -1).values
        max_next_q_values[terminated_mask_tensor] = 0.0
        rewards = torch.tensor([replay.reward for replay in replay_batch], device=self.device)

        # bellman equation
        optimal_values = rewards + self.gamma * max_next_q_values

        loss = F.huber_loss(q_values, optimal_values, delta=1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()