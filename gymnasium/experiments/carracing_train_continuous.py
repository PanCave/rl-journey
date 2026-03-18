import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from collections import deque
import torch
import numpy as np

from agents.continuous_agent import SACAgent
from networks.continuous_car_racing_cnn import ContinuousCarRacingPolicy, ContinuousCarRacingCritic
from utils.dataclasses import ReplayContinuous
import utils.preprocessing as prep
import utils.checkpoints as chkpts
import utils.batch_sampling as bts
from torch.optim import Adam

BATCH_SIZE = 256
REPLAY_BUFFER_RESET_STEPS = 1000

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'  # SCHMUTZ
else:
    device = 'cpu'

# 0: steering, -1 is full left, +1 is full right
# 1: gas
# 2: braking
env = gym.make('CarRacing-v3', render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=True, continuous=True, max_episode_steps=100)

NUM_EPISODES = 10_000
NUM_TIMESTEPS = 10_000
MAX_REPLAY_BUFFER_LENGTH = 10_000
EPISODE_SAVE_RATE = 25
EXPERIMENT_NAME = 'sac_cpu_haltpunish/'
CHECKPOINTS_PARENT_DIRECTORY = 'gymnasium/checkpoints/carracing_sac/'
CHECKPOINTS_SAVE_SUB_DIRECTORY = EXPERIMENT_NAME
CHECKPOINTS_LOAD_SUB_DIRECTORY = 'sac_cpu/'
CHECKPOINTS_SAVE_PATH = CHECKPOINTS_PARENT_DIRECTORY + CHECKPOINTS_SAVE_SUB_DIRECTORY + 'episode_{episode_idx}.pth'
REPEAT_ACTION_NUMBER = 6
STATE_SLICES = (slice(6, -6), slice(None, -12), slice(None, None))

replay_buffer_reset_step_counter = 0

writer = SummaryWriter("gymnasium/runs/carracing_sac/" + EXPERIMENT_NAME)

checkpoint = None
LOAD_EPISODE = 1500
load_checkpoint_path = CHECKPOINTS_PARENT_DIRECTORY + CHECKPOINTS_LOAD_SUB_DIRECTORY + f'episode_{LOAD_EPISODE}.pth'
if os.path.exists(load_checkpoint_path):
    checkpoint = chkpts.load_checkpoint(load_checkpoint_path=load_checkpoint_path)

state_width = 84
state_height = 84
number_of_frames = 4
input_shape = (state_width, state_height, number_of_frames)
output_shape = 3
sac_policy = ContinuousCarRacingPolicy(input_shape=input_shape, action_dim=output_shape)
critic_1_network = ContinuousCarRacingCritic(input_shape=input_shape, action_dim=3)
critic_2_network = ContinuousCarRacingCritic(input_shape=input_shape, action_dim=3)

critic_1_optimizer = Adam(critic_1_network.parameters(), lr=0.0001)
critic_2_optimizer = Adam(critic_2_network.parameters(), lr=0.0001)
policy_optimizer = Adam(sac_policy.parameters(), lr=0.0001)
agent = SACAgent(
    env=env,
    policy_network=sac_policy,
    critic_1_network=critic_1_network,
    critic_2_network=critic_2_network,
    alpha=0.2,
    tau=0.95,
    gamma=0.995,
    critic_1_optimizer=critic_1_optimizer,
    critic_2_optimizer=critic_2_optimizer,
    policy_optimizer=policy_optimizer,
    device=device,
    action_range_mins=np.array([-1, 0, 0]),
    action_range_maxs=np.array([1, 1, 1])
)
empty_state = torch.zeros(state_width, state_height)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
global_step_counter = 0
episode_start_number = 0

if checkpoint is not None:
    assert agent.critic_1_network is ContinuousCarRacingCritic
    assert agent.critic_2_network is ContinuousCarRacingCritic
    # TODO: Reactivate, once lr is properly set in checkpoint
    # agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
    agent.critic_1_network.load_state_dict(checkpoint['critic_1_network_state_dict'])
    agent.critic_2_network.load_state_dict(checkpoint['critic_2_network_state_dict'])
    episode_start_number = checkpoint['episode_idx']


for episode_idx in range(episode_start_number, NUM_EPISODES):
    
    state, info = env.reset()

    sum_episode_reward = 0
    sum_policy_episode_loss = 0
    sum_critic_1_episode_loss = 0
    sum_critic_2_episode_loss = 0
    episode_step_counter = 0

    print(f'Episode {episode_idx}')
    writer.add_scalar("Alpha", agent.alpha, episode_idx)

    grayscaled = prep.convert_to_grayscale(state, slices=STATE_SLICES)
    states_queue = deque(maxlen=number_of_frames, iterable=[empty_state]*(number_of_frames - 1) + [grayscaled])
    
    for timestep in range(NUM_TIMESTEPS):
        global_step_counter += 1

        grayscaled_state = prep.convert_to_grayscale(state=state, slices=STATE_SLICES)
        states_queue.append(grayscaled_state)
        agent_state = prep.deque_to_tensor(states_queue)
        action = agent.select_action(agent_state)

        repeat_action_reward = 0
        for _ in range(REPEAT_ACTION_NUMBER):
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_step_counter += 1
            repeat_action_reward += float(reward)

            next_grayscaled_state = prep.convert_to_grayscale(state=next_state, slices=STATE_SLICES)
            states_queue.append(next_grayscaled_state)

            if truncated or terminated:
                break

            state = next_state
        

        next_agent_state = prep.deque_to_tensor(states_queue)

        sum_episode_reward += repeat_action_reward
        
        experience = ReplayContinuous(agent_state, action, repeat_action_reward, next_agent_state, terminated)
        replay_buffer.append(experience)
        
        if len(replay_buffer) >= BATCH_SIZE and timestep % 1 == 0:
            batch = bts.sample_continuous_with_high_rewards_prioritized(replay_buffer=replay_buffer, number_of_samples=BATCH_SIZE)
            loss = agent.train(batch)
            sum_policy_episode_loss += loss['policy_loss']
            sum_critic_1_episode_loss += loss['critic_1_loss']
            sum_critic_2_episode_loss += loss['critic_2_loss']
        
        if terminated or truncated:
            break
        
        state = next_state

    mean_episode_reward = sum_episode_reward / episode_step_counter
    writer.add_scalar("Summed Reward per Episode", sum_episode_reward, episode_idx)
    writer.add_scalar("Mean Reward per Episode", mean_episode_reward, episode_idx)
    writer.add_scalar("Summed Policy Loss per Episode", sum_policy_episode_loss, episode_idx)
    writer.add_scalar("Summed Critic 1 Loss per Episode", sum_critic_1_episode_loss, episode_idx)
    writer.add_scalar("Summed Critic 2 Loss per Episode", sum_critic_2_episode_loss, episode_idx)
    writer.add_scalar("Episode Step Counter", episode_step_counter, episode_idx)

    if episode_idx > 0 and episode_idx % EPISODE_SAVE_RATE == 0:
        chkpts.save_sac_checkpoint(
            agent=agent, 
            episode_idx=episode_idx, 
            save_checkpoint_path_str=CHECKPOINTS_SAVE_PATH.format(episode_idx=episode_idx)
        )
    

env.close()