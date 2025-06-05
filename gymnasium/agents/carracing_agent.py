import random

import gymnasium as gym
import numpy as np
from copy import deepcopy
from typing import List
import torch.nn.functional as F
import torch.nn as nn
import torch

from utils.dataclasses import Replay

class DQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(DQN, self).__init__()
        # CNN layers to process the image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[2], 16, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the size of flattened features
        conv_output_size = self._get_conv_output(input_shape)

        # Fully connected layers (3 hidden layers with 128 neurons each)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.Linear(512, 128),
            nn.Linear(128, action_dim)
        )

    def _get_conv_output(self, shape) -> int:
        # Forward pass with dummy input to get output shape
        bs = 1
        dummy_data = torch.zeros(bs, shape[2], shape[0], shape[1])
        x = self.conv_layers(dummy_data)

        return x.flatten(1).size(1)

    def forward(self, tensor_input: torch.Tensor):
        # Ensure input has the right format (batch_size, channels, height, width)
        # Original shape: (batch_size, height, width, channels)
        # No longer needed, as we now permute in preprocessing
        #tensor_input = tensor_input.permute(0, 3, 1, 2)

        # Normalize pixel values to [0, 1]
        tensor_input = tensor_input.float() / 255.0

        # CNN layers
        tensor_input = self.conv_layers(tensor_input)

        # Flatten
        tensor_input = tensor_input.flatten(1)

        # Fully connected layers
        tensor_input = self.fc_layers(tensor_input)

        return tensor_input


class MichaelSchumacherDiscrete:
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
            done_mask = np.array([replay.done for replay in replay_batch])
            done_mask_tensor = torch.tensor(done_mask, device=self.device, dtype=torch.bool)
            next_q_values = self.target_network.forward(next_states_tensor)

        max_next_q_values = torch.max(
            input = next_q_values,
            dim = -1).values
        max_next_q_values[done_mask_tensor] = 0.0
        rewards = torch.tensor([replay.reward for replay in replay_batch], device=self.device)

        # bellman equation
        optimal_values = rewards + self.gamma * max_next_q_values

        loss = F.huber_loss(q_values, optimal_values, delta=1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()