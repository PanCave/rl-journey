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
        gamma: float,
        alpha: float,
        tau: float,
        critic_1_optimizer: torch.optim.Optimizer | None,
        critic_2_optimizer: torch.optim.Optimizer | None,
        policy_optimizer: torch.optim.Optimizer | None,
        device: torch.device,
        action_range_mins: NDArray,
        action_range_maxs: NDArray,
        alpha_optimizer: torch.optim.Optimizer | None = None,
        target_entropy: float | None = None
    ) -> None:
        assert len(action_range_mins) == len(action_range_maxs)
        self.action_dim = len(action_range_mins)

        self.action_range_mins = action_range_mins
        action_range_diffs = action_range_maxs - action_range_mins
        self.scale_factor = 0.5 * action_range_diffs # 0.5 because of tanh
        self.action_range_mins_tensor = torch.tensor(action_range_mins, dtype=torch.float32, device=device)
        self.scale_factor_tensor = torch.tensor(self.scale_factor, dtype=torch.float32, device=device)

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
        self.gamma = gamma
        self.tau = tau

        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = alpha_optimizer
        self.target_entropy = target_entropy if target_entropy is not None else -float(self.action_dim)

        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer
        self.policy_optimizer = policy_optimizer
        self.policy_network.to(device=self.device)
        if self.critic_1_network is not None:
            self.critic_1_network.to(device=self.device)
        if self.critic_2_network is not None:
            self.critic_2_network.to(device=self.device)

    def _map_to_target_range(self, values: NDArray) -> NDArray:
        return self.scale_factor * (values + 1) + self.action_range_mins

    def _map_tensor_to_target_range(self, values: torch.Tensor) -> torch.Tensor:
        return self.scale_factor_tensor * (values + 1) + self.action_range_mins_tensor

    def select_action(
        self,
        state: torch.Tensor,
        inference_only: bool = False
    ) -> NDArray:
        with torch.no_grad():
            self.policy_network.eval()
            state = torch.unsqueeze(state, 0).to(device=self.device)
            # nn should output mu and log_sigma
            mu_sigma_values = self.policy_network(state)
            mu_sigma_values = mu_sigma_values.squeeze().detach().cpu().numpy()
            mu = mu_sigma_values[:self.action_dim]
            
        if inference_only:
            action = np.tanh(mu)
            return self._map_to_target_range(action)
        else:
            log_sigma = mu_sigma_values[self.action_dim:]
            log_sigma = np.clip(log_sigma, -20, 2)
            sigma = np.exp(log_sigma)
            #action_dim_tensor = torch.tensor(self.action_dim, device=self.device)
            e = np.random.randn(self.action_dim)
            u = mu + sigma * e
            action = np.tanh(u)
            
            return self._map_to_target_range(action)
        
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
        # y(r, s', d) = r + gamma*(1-terminated)*(min_{i=1,2} Q_{phi_targ, i}(s', a'*) - alpha * log(pi_theta(a'*|s')))
        # y(r, s', d): Huber Loss
        # r: reward
        # gamma: discount factor
        # terminated: terminated flag
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
        actions_np = np.asarray([replay.action for replay in replay_batch], dtype=np.float32)
        rewards_np = np.asarray([replay.reward for replay in replay_batch], dtype=np.float32)
        next_states = torch.stack([replay.next_state for replay in replay_batch]).to(device=self.device)
        terminated_np = np.asarray([replay.terminated for replay in replay_batch], dtype=np.float32)
        actions = torch.as_tensor(actions_np, device=self.device)
        rewards = torch.as_tensor(rewards_np, device=self.device)
        terminated = torch.as_tensor(terminated_np, device=self.device)
        # Ensuring correct shapes
        rewards = rewards.unsqueeze(-1)
        terminated = terminated.unsqueeze(-1).float()
        q_values_1: torch.Tensor = self.critic_1_network(states, actions)
        q_values_1 = q_values_1.view(-1, 1)
        q_values_2: torch.Tensor = self.critic_2_network(states, actions)
        q_values_2 = q_values_2.view(-1, 1)

        with torch.no_grad():
            ## nn should output mu and log_sigma
            next_actions_policy: torch.Tensor = self.policy_network(next_states)
            mu = next_actions_policy[:, :self.action_dim]
            log_sigma = next_actions_policy[:, self.action_dim:]
            log_sigma = torch.clamp(log_sigma, min=-20, max=2)
            sigma = torch.exp(log_sigma)
            pi_theta = torch.distributions.Normal(mu, sigma)
            # e = torch.randn((len(replay_batch), self.action_dim))
            # u = mu + sigma * e
            m_sample = pi_theta.rsample() # shape: (batch_size, action_dim)
            next_actions_sampled = torch.tanh(m_sample)
            next_actions_mapped = self._map_tensor_to_target_range(next_actions_sampled)

            # Calculate min_{i=1,2}  Q_{phi_targ, i}(s', a'*)
            target_1_q_values: torch.Tensor = self.target_1_network(next_states, next_actions_mapped)
            target_2_q_values: torch.Tensor = self.target_2_network(next_states, next_actions_mapped)
            minimun_q_values = torch.minimum(target_1_q_values, target_2_q_values)

            # Calculate log(pi_theta(a'*|s'))
            logp_basic = pi_theta.log_prob(m_sample).sum(dim=-1, keepdim=True) - torch.log(1 - next_actions_sampled.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

            # y(r, s', d)
            alpha = self.log_alpha.exp()
            y_values = rewards + (1 - terminated) * self.gamma * (minimum_q_values - alpha * logp_basic)
        
        assert rewards.ndim == 2 and rewards.shape[1] == 1
        assert terminated.ndim == 2 and terminated.shape[1] == 1
        assert q_values_1.shape == q_values_2.shape == rewards.shape == terminated.shape
        assert logp_basic.shape == rewards.shape

        huber_value_loss_1 = F.huber_loss(q_values_1, y_values, delta=1)
        huber_value_loss_2 = F.huber_loss(q_values_2, y_values, delta=1)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        huber_value_loss_1.backward()
        huber_value_loss_2.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Policy loss function:
        # a* = pi_theta(.|s): predicted action for s
        # log(pi_theta(a*|s)): entropy term
        actions_policy: torch.Tensor = self.policy_network(states)
        mu = actions_policy[:, :self.action_dim]
        log_sigma = actions_policy[:, self.action_dim:]
        log_sigma = torch.clamp(log_sigma, min=-20, max=2)
        sigma = torch.exp(log_sigma)
        pi_theta = torch.distributions.Normal(mu, sigma)
        # e = torch.randn((len(replay_batch), self.action_dim))
        # u = mu + sigma * e
        m_sample = pi_theta.rsample() # shape: (batch_size, action_dim)
        actions_sampled = torch.tanh(m_sample)
        actions_mapped = self._map_tensor_to_target_range(actions_sampled)

        # Calculate min_{i=1,2}  Q_{phi_targ, i}(s, a*)
        critic_parameters = list(self.critic_1_network.parameters()) + list(self.critic_2_network.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)

        critic_1_q_values = self.critic_1_network(states, actions_mapped)
        critic_2_q_values = self.critic_2_network(states, actions_mapped)
        minimun_q_values = torch.minimum(critic_1_q_values, critic_2_q_values)

        # Calculate log(pi_theta(a*|s))
        logp_basic = pi_theta.log_prob(m_sample).sum(dim=-1, keepdim=True) - torch.log(1 - actions_sampled.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        alpha = self.log_alpha.exp().detach()
        policy_loss = (alpha * logp_basic - minimum_q_values).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        # soft-update of target networks
        self.soft_update(self.target_1_network, self.critic_1_network)
        self.soft_update(self.target_2_network, self.critic_2_network)

        metrics = {
            'critic_1_loss': float(huber_value_loss_1.detach().item()),
            'critic_2_loss': float(huber_value_loss_2.detach().item()),
            'policy_loss': float(policy_loss.detach().item()),
            'alpha': float(self.log_alpha.exp().detach().item())
        }
        if alpha_loss is not None:
            metrics['alpha_loss'] = float(alpha_loss.detach().item())
        return metrics
