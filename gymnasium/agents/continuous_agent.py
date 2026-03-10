from typing import List
import torch.nn as nn
import gymnasium as gym
import torch
import numpy as np
from copy import deepcopy

from numpy.typing import NDArray
import torch.nn.functional as F

from utils.dataclasses import ReplayContinuous

class SACAgent:
    def __init__(
        self,
        env: gym.Env,
        policy_network: nn.Module,
        critic_1_network: nn.Module | None,
        critic_2_network: nn.Module | None,
        action_dim: int,
        gamma: float,
        alpha: float,
        tau: float,
        critic_1_optimizer: torch.optim.Optimizer | None,
        critic_2_optimizer: torch.optim.Optimizer | None,
        policy_optimizer: torch.optim.Optimizer | None,
        device: torch.device | str
    ) -> None:
        self.env = env
        self.policy_network = policy_network
        self.critic_1_network = critic_1_network
        self.target_1_network = deepcopy(critic_1_network)
        self.device = device
        if self.target_1_network is not None:
            self.target_1_network.to(device=self.device)
            for p in self.target_1_network.parameters():
                p.requires_grad_(False)
        self.critic_2_network = critic_2_network
        self.target_2_network = deepcopy(critic_2_network)
        if self.target_2_network is not None:
            self.target_2_network.to(device=self.device)
            for p in self.target_2_network.parameters():
                p.requires_grad_(False)
        self.target_net_update_step_counter = 0
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer
        self.policy_optimizer = policy_optimizer
        self.policy_network.to(device=self.device)
        #self.target_network.to(device=self.device)

    def select_action(
        self,
        state: torch.Tensor,
        inference_only: bool = False
    ) -> NDArray:
        with torch.no_grad():
            self.policy_network.eval()
            state = torch.unsqueeze(state, 0).to(device=self.device)
            # nn should output mu and log_sigma
            mu_sigma_values = self.policy_network.forward(state)
            mu_sigma_values = mu_sigma_values.squeeze().detach().cpu().numpy()
            mu = mu_sigma_values[:self.action_dim]
            
        if inference_only:
            return np.tanh(mu)
        else:
            log_sigma = mu_sigma_values[self.action_dim:]
            sigma = np.exp(log_sigma)
            #action_dim_tensor = torch.tensor(self.action_dim, device=self.device)
            e = np.random.randn(self.action_dim)
            u = mu + sigma * e
            action = np.tanh(u)
            
            return action
        
    @torch.no_grad()
    def soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
        
    
    def train(self, replay_batch: List[ReplayContinuous]) -> dict[str, float]:
        assert isinstance(self.critic_1_network, nn.Module)
        assert isinstance(self.critic_2_network, nn.Module)
        assert isinstance(self.target_1_network, nn.Module)
        assert isinstance(self.target_2_network, nn.Module)
        assert isinstance(self.policy_optimizer, torch.optim.Optimizer)
        assert isinstance(self.critic_1_optimizer, torch.optim.Optimizer)
        assert isinstance(self.critic_2_optimizer, torch.optim.Optimizer)
        # Value loss function:
        # y(r, s', d) = r + gamma*(1-done)*(min_{i=1,2} Q_{phi_targ, i}(s', a'*) - alpha * log(pi_theta(a'*|s')))
        # y(r, s', d): MSE
        # r: reward
        # gamma: discount factor
        # done: done flag
        # phi_{targ, i}: i-th target network
        # a'*: pi_theta(.|s'): predicted action for s'
        # Q_{phi_targ, i}(s', a'*): q-value of the i-th target network
        # log(pi_theta(a'*|s')): entropy term

        # Set networks to train mode
        self.policy_network.train()
        self.critic_1_network.train()
        self.critic_2_network.train()

        # Calculate a'*
        states = torch.stack([replay.state for replay in replay_batch]).to(device=self.device)
        actions = torch.tensor([replay.action for replay in replay_batch], device=self.device, dtype=torch.float32)
        rewards = torch.tensor([replay.reward for replay in replay_batch], device=self.device, dtype=torch.float32)
        next_states = torch.stack([replay.next_state for replay in replay_batch]).to(device=self.device)
        done = torch.tensor([replay.done for replay in replay_batch], device=self.device, dtype=torch.float32)
        # Ensuring correct shapes
        rewards = rewards.unsqueeze(-1)
        done = done.unsqueeze(-1).float()
        q_values_1 = self.critic_1_network.forward(states, actions)
        q_values_2 = self.critic_2_network.forward(states, actions)

        with torch.no_grad():
            ## nn should output mu and log_sigma
            next_actions_policy: torch.Tensor = self.policy_network.forward(next_states)
            mu = next_actions_policy[:, :self.action_dim]
            log_sigma = next_actions_policy[:, self.action_dim:]
            log_sigma = torch.clamp(log_sigma, min=-20, max=2)
            sigma = torch.exp(log_sigma)
            pi_theta = torch.distributions.Normal(mu, sigma)
            # e = torch.randn((len(replay_batch), self.action_dim))
            # u = mu + sigma * e
            m_sample = pi_theta.rsample() # shape: (batch_size, action_dim)
            next_actions_sampled = torch.tanh(m_sample)

            # Calculate min_{i=1,2}  Q_{phi_targ, i}(s', a'*)
            target_1_q_values: torch.Tensor = self.target_1_network.forward(next_states, next_actions_sampled)
            target_2_q_values: torch.Tensor = self.target_2_network.forward(next_states, next_actions_sampled)
            minimun_q_values = torch.minimum(target_1_q_values, target_2_q_values)

            # Calculate log(pi_theta(a'*|s'))
            logp_basic = pi_theta.log_prob(m_sample).sum(dim=-1, keepdim=True) - torch.log(1 - next_actions_sampled.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

            # y(r, s', d)
            y_values = rewards + (1 - done) * self.gamma * (minimun_q_values - self.alpha * logp_basic)

        mse_value_loss_1 = F.huber_loss(q_values_1, y_values, delta=1)
        mse_value_loss_2 = F.huber_loss(q_values_2, y_values, delta=1)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        mse_value_loss_1.backward()
        mse_value_loss_2.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Policy loss function:
        # a* = pi_theta(.|s): predicted action for s
        # log(pi_theta(a*|s)): entropy term
        actions_policy: torch.Tensor = self.policy_network.forward(states)
        mu = actions_policy[:, :self.action_dim]
        log_sigma = actions_policy[:, self.action_dim:]
        log_sigma = torch.clamp(log_sigma, min=-20, max=2)
        sigma = torch.exp(log_sigma)
        pi_theta = torch.distributions.Normal(mu, sigma)
        # e = torch.randn((len(replay_batch), self.action_dim))
        # u = mu + sigma * e
        m_sample = pi_theta.rsample() # shape: (batch_size, action_dim)
        actions_sampled = torch.tanh(m_sample)

        # Calculate min_{i=1,2}  Q_{phi_targ, i}(s, a*)
        critic_1_q_values: torch.Tensor = self.critic_1_network.forward(states, actions_sampled)
        critic_2_q_values: torch.Tensor = self.critic_2_network.forward(states, actions_sampled)
        minimun_q_values = torch.minimum(critic_1_q_values, critic_2_q_values)

        # Calculate log(pi_theta(a*|s))
        logp_basic = pi_theta.log_prob(m_sample).sum(dim=-1, keepdim=True) - torch.log(1 - actions_sampled.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        policy_loss = (self.alpha * logp_basic - minimun_q_values).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # soft-update of target networks
        self.soft_update(self.target_1_network, self.critic_1_network)
        self.soft_update(self.target_2_network, self.critic_2_network)

        return {
            'critic_loss_1': float(mse_value_loss_1.detach().item()),
            'critic_loss_2': float(mse_value_loss_2.detach().item()),
            'policy_loss': float(policy_loss.detach().item())
        }