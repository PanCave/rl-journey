import random

import gymnasium as gym
import numpy as np
from copy import deepcopy
from typing import List
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.dataclasses import Replay

class DQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(DQN, self).__init__()
        # CNN layers to process the image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the size of flattened features
        conv_output_size = self._get_conv_output(input_shape)

        # Fully connected layers (3 hidden layers with 128 neurons each)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
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
        tensor_input = tensor_input.permute(0, 3, 1, 2)

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
        gamma: float,
        summary_writer: SummaryWriter,
        device: torch.device
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
        self.target_net_update_step_counter = 0
        self.summary_writer = summary_writer
        
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters())
        self.device = device
        self.policy_network.to(device=self.device)
        self.target_network.to(device=self.device)

    def select_action(
        self,
        state: torch.Tensor
    ) -> int:
        value = random.random()
        if value <= self.epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            with torch.no_grad():
                self.policy_network.eval()
                state = torch.unsqueeze(state, 0)
                q_values = self.policy_network.forward(state)
                action = int(torch.argmax(q_values))
                return action
            
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)
        if self.epsilon < 0.3 * self.epsilon_init:
            self.epsilon_init *= 0.8
            self.epsilon = self.epsilon_init

    def train(self,
              replay_batch: List[Replay],
              global_step: int) -> None:
        # Update target network after n steps
        self.target_net_update_step_counter += 1
        if (self.target_net_update_step_counter == self.num_target_update_steps):
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.target_net_update_step_counter = 0
            
        self.policy_network.train()
        
        states = np.array([replay.state for replay in replay_batch])
        states_tensor = torch.tensor(states, device=self.device)
        q_values_batch = self.policy_network(states_tensor)
        row_indices = torch.arange(q_values_batch.size(0), device=self.device)
        actions = [replay.action for replay in replay_batch]
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
        q_values = q_values_batch[row_indices, actions_tensor]

        with torch.no_grad():
            done_mask = np.array([replay.done for replay in replay_batch])
            done_mask_tensor = torch.tensor(done_mask, device=self.device, dtype=torch.bool)
            next_states = np.array([replay.next_state for replay in replay_batch])
            next_states_tensor = torch.tensor(next_states, device=self.device)
            next_q_values = self.target_network(next_states_tensor)
            max_next_q_values = torch.max(
                input = next_q_values,
                dim = -1).values
            max_next_q_values[done_mask_tensor] = 0.0        
            rewards = torch.tensor([replay.reward for replay in replay_batch], device=self.device)
            optimal_values = rewards + self.gamma * max_next_q_values

        loss = F.mse_loss(q_values, optimal_values)      
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.summary_writer.add_scalar("Loss / Batch", loss.item(), global_step=global_step)