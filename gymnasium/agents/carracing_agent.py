import random

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
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
            nn.Linear(512, 128),
            nn.Linear(128, 64),
            nn.Linear(64, action_dim)
        )
        
    def _get_conv_output(self, shape) -> int:
        # Forward pass with dummy input to get output shape
        bs = 1
        dummy_data = torch.zeros(bs, shape[2], shape[0], shape[1])
        x = self.conv_layers(dummy_data)
        
        return x.flatten(1).size(1)
    
    def forward(self, x):
        # Ensure input has the right format (batch_size, channels, height, width)
        # Original shape: (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)
        
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        
        # CNN layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x
    

class MichaelSchumacherDiscrete:
    def __init__(
        self,
        env: gym.Env,
        num_target_update_steps: int,
        policy_network: nn.Module,
        epsilon_init: float,
        epsilon_min: float,
        delta: float,
        gamma: float
    ) -> None:
        self.env = env
        self.num_target_update_steps = num_target_update_steps
        self.policy_network = policy_network
        self.target_net = deepcopy(policy_network)
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_init
        self.delta = delta
        self.gamma = gamma
        self.target_net_update_step_counter = 0
        
    def select_action(
        self,
        state: np.ndarray
    ) -> int:
        value = random.random()
        if value >= self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.policy_network.forward(state)
            return int(np.argmax(q_values))
        
    def train(self,
              replay_batch: List[Replay]) -> None:
        # Update target network after n steps
        self.target_net_update_step_counter += 1
        if (self.target_net_update_step_counter == self.num_target_update_steps):
            self.target_net = deepcopy(self.policy_network)
            self.target_net_update_step_counter = 0
            
        
        q_max = self.policy_network.forward()
        
        
        loss = F.mse_loss(q_values, max_next_q_values)