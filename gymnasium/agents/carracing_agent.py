import random

import gymnasium as gym
import numpy as np

from copy import deepcopy

class DQN:
    def __init__(self):
        pass
    
    def forward(self, state: np.ndarray) -> np.ndarray: # return type is list of q values
        pass
    

class MichaelSchumacherDiscrete:
    def __init__(
        self,
        env: gym.Env,
        num_target_update_steps: int,
        policy_network: DQN,
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
        self.step_counter = 0
        
    def get_action(
        self,
        state: np.ndarray
    ) -> int:
        value = random.random()
        if value >= self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.policy_network.forward(state)
            return int(np.argmax(q_values))