import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from collections import deque
import torch

from agents.discrete_agent import DiscreteAgent
from networks.dk_dqn import DKDQN
from utils.dataclasses import Replay
import utils.preprocessing as prep
import utils.checkpoints as chkpts
import utils.batch_sampling as bts
from torch.optim import Adam
import ale_py

gym.register_envs(ale_py)

BATCH_SIZE = 256
REPLAY_BUFFER_RESET_STEPS = 1000

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'  # SCHMUTZ
else:
    device = 'cpu'

env = gym.make('ALE/DonkeyKong-v5', render_mode='rgb_array', obs_type='grayscale')

NUM_EPISODES = 10_000
NUM_TIMESTEPS = 10_000
MAX_REPLAY_BUFFER_LENGTH = 10_000
EPISODE_SAVE_RATE = 25
EXPERIMENT_NAME = 'julius_dk/'
CHECKPOINTS_PARENT_DIRECTORY = 'gymnasium/checkpoints/julius_dk/'
CHECKPOINTS_SAVE_SUB_DIRECTORY = EXPERIMENT_NAME
CHECKPOINTS_LOAD_SUB_DIRECTORY = 'julius_dk/'
CHECKPOINTS_SAVE_PATH = CHECKPOINTS_PARENT_DIRECTORY + CHECKPOINTS_SAVE_SUB_DIRECTORY + 'episode_{episode_idx}.pth'
REPEAT_ACTION_NUMBER = 6
STATE_SLICES = (slice(None, None), slice(None, None), slice(None, None))

replay_buffer_reset_step_counter = 0

writer = SummaryWriter("gymnasium/runs/julius_dk/" + EXPERIMENT_NAME)

checkpoint = None
LOAD_EPISODE = -1
load_checkpoint_path = CHECKPOINTS_PARENT_DIRECTORY + CHECKPOINTS_LOAD_SUB_DIRECTORY + f'episode_{LOAD_EPISODE}.pth'
if os.path.exists(load_checkpoint_path):
    checkpoint = chkpts.load_checkpoint(load_checkpoint_path=load_checkpoint_path)

state_width = 160
state_height = 210
number_of_frames = 4
input_shape = (state_height, state_width, number_of_frames)
output_shape = 18
dqn = DKDQN(input_shape=input_shape, action_dim=output_shape)
optimizer = Adam(dqn.parameters(), lr=0.001)
agent = DiscreteAgent(
    env=env,
    num_target_update_steps=2000,
    epsilon_init=1,    # Startwert für Epsilon
    epsilon_min=0.01, # Minimaler Epsilon-Wert
    epsilon_decay_rate=0.995,      # Abnahmerate von Epsilon
    gamma=0.95,          # Discount-Faktor
    optimizer=optimizer,
    device=device,
    policy_network=dqn
)
empty_state = torch.zeros(state_height, state_width)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
global_step_counter = 0
episode_start_number = 0

if checkpoint is not None:
    # TODO: Reactivate, once lr is properly set in checkpoint
    # agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    episode_start_number = checkpoint['episode_idx']


for episode_idx in range(episode_start_number, NUM_EPISODES):
    
    state, info = env.reset()

    non_positive_reward_counter = 0
    sum_episode_reward = 0
    sum_episode_loss = 0
    episode_step_counter = 0

    print(f'Episode {episode_idx}, Epsilon: {agent.epsilon}')
    writer.add_scalar("Epsilon", agent.epsilon, episode_idx)

    grayscaled = prep.convert_to_tensor(state, device=device)
    states_queue = deque(maxlen=number_of_frames, iterable=[empty_state]*(number_of_frames - 1) + [grayscaled])

    for _ in range(50):
        env.step(0)
    
    for timestep in range(NUM_TIMESTEPS):
        global_step_counter += 1

        grayscaled_state = prep.convert_to_tensor(state=state, device=device)
        states_queue.append(grayscaled_state)
        agent_state = prep.deque_to_tensor(states_queue)
        action = agent.select_action(agent_state)

        repeat_action_reward = 0
        for _ in range(REPEAT_ACTION_NUMBER):
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_step_counter += 1
            repeat_action_reward += reward

            next_grayscaled_state = prep.convert_to_tensor(state=next_state, device=device)
            states_queue.append(next_grayscaled_state)

            if truncated or terminated:
                break

            state = next_state        

        next_agent_state = prep.deque_to_tensor(states_queue)

        sum_episode_reward += repeat_action_reward
        
        experience = Replay(agent_state, action, repeat_action_reward, next_agent_state, terminated or truncated)
        replay_buffer.append(experience)
        
        if len(replay_buffer) >= BATCH_SIZE and timestep % 4 == 0:
            batch = bts.sample_with_high_rewards_prioritized(replay_buffer=replay_buffer, number_of_samples=BATCH_SIZE)
            loss = agent.train(batch)
            sum_episode_loss += loss
        
        if terminated or truncated:
            break
        
        state = next_state

    agent.update_epsilon()

    mean_episode_reward = sum_episode_reward / episode_step_counter
    writer.add_scalar("Summed Reward per Episode", sum_episode_reward, episode_idx)
    writer.add_scalar("Mean Reward per Episode", mean_episode_reward, episode_idx)
    writer.add_scalar("Summed Loss per Episode", sum_episode_loss, episode_idx)
    writer.add_scalar("Episode Step Counter", episode_step_counter, episode_idx)

    if episode_idx > 0 and episode_idx % EPISODE_SAVE_RATE == 0:
        chkpts.save_checkpoint(
            agent=agent, 
            episode_idx=episode_idx, 
            save_checkpoint_path_str=CHECKPOINTS_SAVE_PATH.format(episode_idx=episode_idx)
        )

    if agent.epsilon <= agent.epsilon_init * 0.2:
        agent.epsilon_init *= 0.7
        agent.reset_epsilon()
    

env.close()