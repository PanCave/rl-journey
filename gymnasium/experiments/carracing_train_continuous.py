import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from clearml import Task

from collections import deque
import torch
import torch.nn as nn
import numpy as np

from agents.continuous_agent import SACAgent
from networks.continuous_car_racing_cnn import ContinuousCarRacingPolicy, ContinuousCarRacingCritic
from utils.dataclasses import ReplayContinuous
import utils.preprocessing as prep
import utils.checkpoints as chkpts
import utils.batch_sampling as bts
from torch.optim import Adam

task = Task.init(
    project_name="SAC CarRacing",
    task_name="SAC alpha tuning"
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BATCH_SIZE = 256
REPLAY_BUFFER_RESET_STEPS = 1000

if torch.cuda.is_available():
    device = torch.device('cuda')
# elif torch.mps.is_available():
#     device = 'mps'  # SCHMUTZ
else:
    device = torch.device('cpu')

print(device)
# 0: steering, -1 is full left, +1 is full right
# 1: gas
# 2: braking
env = gym.make('CarRacing-v3', render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=True, continuous=True)

NUM_EPISODES = 10_000
NUM_TIMESTEPS = 10_000
MAX_REPLAY_BUFFER_LENGTH = 10_000
EPISODE_SAVE_RATE = 25
EXPERIMENT_NAME = 'sac_alpha_tuning_1/'
CHECKPOINTS_PARENT_DIRECTORY = 'gymnasium/checkpoints/carracing_sac/'
CHECKPOINTS_SAVE_SUB_DIRECTORY = EXPERIMENT_NAME
CHECKPOINTS_LOAD_SUB_DIRECTORY = 'sac_alpha_tuning_1/'
CHECKPOINTS_SAVE_PATH = CHECKPOINTS_PARENT_DIRECTORY + CHECKPOINTS_SAVE_SUB_DIRECTORY + 'episode_{episode_idx}.pth'
REPEAT_ACTION_NUMBER = 6
STATE_SLICES = (slice(6, -6), slice(None, -12), slice(None, None))

POLICY_LEARNING_RATE = 3e-5
CRITIC_LEARNING_RATE = 4e-5
ALPHA_LEARNING_RATE = 3e-5

replay_buffer_reset_step_counter = 0

logger = task.get_logger()

checkpoint = None
LOAD_EPISODE = -1
load_checkpoint_path = CHECKPOINTS_PARENT_DIRECTORY + CHECKPOINTS_LOAD_SUB_DIRECTORY + f'episode_{LOAD_EPISODE}.pth'
if os.path.exists(load_checkpoint_path):
    print(f"Loading checkpoint {load_checkpoint_path} (Episode {LOAD_EPISODE})")
    checkpoint = chkpts.load_checkpoint(load_checkpoint_path=load_checkpoint_path)
else:
    print("No checkpoint loaded! Restarting training")

state_width = 84
state_height = 84
number_of_frames = 4
input_shape = (state_width, state_height, number_of_frames)
output_shape = 3
policy_network = ContinuousCarRacingPolicy(input_shape=input_shape, action_dim=output_shape)
critic_1_network = ContinuousCarRacingCritic(input_shape=input_shape, action_dim=3)
critic_2_network = ContinuousCarRacingCritic(input_shape=input_shape, action_dim=3)

critic_1_optimizer = Adam(critic_1_network.parameters(), lr=CRITIC_LEARNING_RATE)
critic_2_optimizer = Adam(critic_2_network.parameters(), lr=CRITIC_LEARNING_RATE)
policy_optimizer = Adam(policy_network.parameters(), lr=POLICY_LEARNING_RATE)
agent = SACAgent(
    env=env,
    policy_network=policy_network,
    critic_1_network=critic_1_network,
    critic_2_network=critic_2_network,
    alpha=0.5,
    tau=0.05,
    gamma=0.995,
    critic_1_optimizer=critic_1_optimizer,
    critic_2_optimizer=critic_2_optimizer,
    policy_optimizer=policy_optimizer,
    device=device,
    action_range_mins=np.array([-1, -1, -1]),
    action_range_maxs=np.array([1, 1, 1])
)
alpha_optimizer = Adam([agent.log_alpha], lr=ALPHA_LEARNING_RATE)
agent.alpha_optimizer = alpha_optimizer
empty_state = torch.zeros(state_width, state_height)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
global_step_counter = 0
episode_start_number = 0

if checkpoint is not None:
    assert isinstance(agent.critic_1_network, nn.Module)
    assert isinstance(agent.critic_2_network, nn.Module)
    assert isinstance(agent.target_1_network, nn.Module)
    assert isinstance(agent.target_2_network, nn.Module)
    assert isinstance(agent.policy_optimizer, torch.optim.Optimizer)
    assert isinstance(agent.critic_1_optimizer, torch.optim.Optimizer)
    assert isinstance(agent.critic_2_optimizer, torch.optim.Optimizer)
    
    agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
    agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    agent.critic_1_network.load_state_dict(checkpoint['critic_1_network_state_dict'])
    agent.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
    agent.critic_2_network.load_state_dict(checkpoint['critic_2_network_state_dict'])
    agent.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
    agent.target_1_network.load_state_dict(checkpoint['target_1_network_state_dict'])
    agent.target_2_network.load_state_dict(checkpoint['target_2_network_state_dict'])
    if 'log_alpha' in checkpoint:
        with torch.no_grad():
            agent.log_alpha.copy_(checkpoint['log_alpha'].to(device))
    if 'alpha_optimizer_state_dict' in checkpoint and agent.alpha_optimizer is not None:
        agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
    episode_start_number = checkpoint['episode_idx']

task.connect({
    "BATCH_SIZE": BATCH_SIZE,
    "REPLAY_BUFFER_RESET_STEPS": REPLAY_BUFFER_RESET_STEPS,
    "NUM_EPISODES": NUM_EPISODES,
    "NUM_TIMESTEPS": NUM_TIMESTEPS,
    "MAX_REPLAY_BUFFER_LENGTH": MAX_REPLAY_BUFFER_LENGTH,
    "EPISODE_SAVE_RATE": EPISODE_SAVE_RATE,
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "REPEAT_ACTION_NUMBER": REPEAT_ACTION_NUMBER,
    "LOAD_EPISODE": LOAD_EPISODE,
    "POLICY_LEARNING_RATE": POLICY_LEARNING_RATE,
    "CRITIC_LEARNING_RATE": CRITIC_LEARNING_RATE,
    "ALPHA_LEARNING_RATE": ALPHA_LEARNING_RATE,
    "TARGET_ENTROPY": agent.target_entropy,
    "state_width": state_width,
    "state_height": state_height,
    "number_of_frames": number_of_frames,
})


for episode_idx in range(episode_start_number, NUM_EPISODES):
    
    state, info = env.reset()

    sum_episode_reward = 0
    sum_policy_episode_loss = 0
    sum_critic_1_episode_loss = 0
    sum_critic_2_episode_loss = 0
    sum_alpha_episode_loss = 0
    episode_step_counter = 0
    train_steps = 0

    print(f'Episode {episode_idx}')
    logger.report_scalar("train", "Alpha", value=agent.log_alpha.exp().item(), iteration=episode_idx)

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
            batch = bts.sample_continuous(replay_buffer=replay_buffer, number_of_samples=BATCH_SIZE)
            loss = agent.train(batch)
            train_steps += 1
            sum_policy_episode_loss += loss['policy_loss']
            sum_critic_1_episode_loss += loss['critic_1_loss']
            sum_critic_2_episode_loss += loss['critic_2_loss']
            if 'alpha_loss' in loss:
                sum_alpha_episode_loss += loss['alpha_loss']
        
        if terminated or truncated:
            break
        
        state = next_state

    mean_episode_reward = sum_episode_reward / episode_step_counter
    logger.report_scalar("train", "Alpha", iteration=episode_idx, value=agent.log_alpha.exp().item())
    logger.report_scalar("reward", "Summed Reward per Episode", iteration=episode_idx, value=sum_episode_reward)
    logger.report_scalar("reward", "Mean Reward per Episode", iteration=episode_idx, value=mean_episode_reward)
    logger.report_scalar("summed loss", "Summed Policy Loss", iteration=episode_idx, value=sum_policy_episode_loss)
    logger.report_scalar("summed loss", "Summed Critic 1 Loss", iteration=episode_idx, value=sum_critic_1_episode_loss)
    logger.report_scalar("summed loss", "Summed Critic 2 Loss", iteration=episode_idx, value=sum_critic_2_episode_loss)
    logger.report_scalar("mean loss", "Mean Policy Loss", iteration=episode_idx, value=0 if train_steps == 0 else sum_policy_episode_loss/train_steps)
    logger.report_scalar("mean loss", "Mean Critic 1 Loss", iteration=episode_idx, value=0 if train_steps == 0 else sum_critic_1_episode_loss/train_steps)
    logger.report_scalar("mean loss", "Mean Critic 2 Loss", iteration=episode_idx, value=0 if train_steps == 0 else sum_critic_2_episode_loss/train_steps)
    logger.report_scalar("mean loss", "Mean Alpha Loss", iteration=episode_idx, value=0 if train_steps == 0 else sum_alpha_episode_loss/train_steps)
    logger.report_scalar("episode", "Step Counter", iteration=episode_idx, value=episode_step_counter)

    if episode_idx % EPISODE_SAVE_RATE == 0:
        chkpts.save_sac_checkpoint(
            agent=agent, 
            episode_idx=episode_idx, 
            save_checkpoint_path_str=CHECKPOINTS_SAVE_PATH.format(episode_idx=episode_idx)
        )
    

env.close()