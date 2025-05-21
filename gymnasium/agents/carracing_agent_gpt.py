import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class CNNQNetwork(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(CNNQNetwork, self).__init__()
        # CNN layers to process the image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of flattened features
        conv_output_size = self._get_conv_output(input_shape)
        
        # Fully connected layers (3 hidden layers with 128 neurons each)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.Linear(128, 128),
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

class MichaelSchumacher:
    def __init__(
        self,
        env : gym.Env,
        learning_rate : float = 0.1,
        initial_epsilon : float = 0.1,
        epsilon_decay : float = 0.99,
        epsilon_min : float = 0.01,
        discount_factor : float = 0.95,
        replay_buffer_size : int = 10000,
        batch_size : int = 64,
        target_update_frequency : int = 1000):
        
        self.env = env
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get observation shape and action dimensions from the environment
        obs_shape = env.observation_space.shape  # (96, 96, 3)
        self.action_dim = env.action_space.shape[0]  # 3
        
        # Create Q-networks (main and target)
        self.q_network = CNNQNetwork(obs_shape, self.action_dim).to(self.device)
        self.target_network = CNNQNetwork(obs_shape, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Training step counter
        self.steps = 0
        
    def select_action(self, state):
        """Select an action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            # Explore: random action
            action = np.random.uniform(
                low=self.env.action_space.low, 
                high=self.env.action_space.high, 
                size=self.action_dim
            )
        else:
            # Exploit: best action according to Q-network
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_indices = q_values.argmax(dim=1).item()
            
            # Convert discrete action index to continuous action
            # This is a simplification - for true continuous control, consider using a policy gradient method
            action_low = self.env.action_space.low
            action_high = self.env.action_space.high
            action_range = action_high - action_low
            
            # Map the discrete action index to the continuous action space
            # Each output neuron corresponds to a discretized action
            action = action_low + action_range * (action_indices / (self.action_dim - 1))
            
        return action
        
    def update_epsilon(self):
        """Update epsilon value with decay"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the network using experiences from the replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample random batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        
        # Compute current Q values
        current_q = self.q_network(states)
        
        # Get actions for current states (needed for continuous action spaces)
        # For simplicity, we'll use the action indices
        action_indices = torch.argmax(current_q, dim=1)
        
        # Get target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = torch.max(next_q, dim=1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * max_next_q
        
        # Compute loss (MSE between current Q and target Q for the taken actions)
        loss = F.mse_loss(torch.sum(current_q * F.one_hot(action_indices, self.action_dim), dim=1), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
    def learn(self, num_episodes):
        """Main training loop"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.remember(state, action, reward, next_state, done)
                self.train()
                
                state = next_state
                episode_reward += reward
                
            self.update_epsilon()
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {episode_reward}, Epsilon: {self.epsilon:.4f}")