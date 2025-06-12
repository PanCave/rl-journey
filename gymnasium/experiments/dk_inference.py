import sys
import os
from collections import deque

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.checkpoints as chkpts
from agents.discrete_agent import DiscreteAgent
from networks.dk_dqn import DKDQN
import utils.preprocessing as prep
import ale_py

gym.register_envs(ale_py)

# Create gifs directory if it doesn't exist
VIDEO_DIRECTORY = 'gymnasium/videos/'
os.makedirs(VIDEO_DIRECTORY, exist_ok=True)

LOAD_EPISODE = 525
CHECKPOINTS_DIRECTORY = 'gymnasium/checkpoints/julius_dk/'
EXPERIMENT_NAME = 'master_lrschedule'
CHECKPOINT_PATH = CHECKPOINTS_DIRECTORY + EXPERIMENT_NAME + f'/episode_{LOAD_EPISODE}.pth'
checkpoint = chkpts.load_checkpoint(CHECKPOINT_PATH)

episode_trigger = lambda t: True
env = gym.make('ALE/DonkeyKong-v5', render_mode='human', obs_type='rgb', max_episode_steps=-1)
#env = RecordVideo(env, video_folder=VIDEO_DIRECTORY + EXPERIMENT_NAME, name_prefix=EXPERIMENT_NAME, fps=60, episode_trigger=episode_trigger, disable_logger=True)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'  # SCHMUTZ
else:
    device = 'cpu'

state_width = 210
state_height = 160
number_of_frames = 4
input_shape = (state_height, state_width, number_of_frames)
output_shape = 18
dqn = DKDQN(input_shape=input_shape, action_dim=output_shape)
optimizer = torch.optim.Adam(dqn.parameters())
agent = DiscreteAgent(
    env=env,
    num_target_update_steps=100,
    epsilon_init=1,    # Startwert für Epsilon
    epsilon_min=0.1, # Minimaler Epsilon-Wert
    epsilon_decay_rate=0.999,      # Abnahmerate von Epsilon
    gamma=0.9,          # Discount-Faktor
    optimizer=optimizer,
    device=device,
    policy_network=dqn
)

empty_state = torch.zeros(state_height, state_width)
states_queue = deque(maxlen=number_of_frames, iterable=[empty_state] * 3)

if checkpoint:
    agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])

for episode_idx in range(20):
    state, _ = env.reset()
    non_positive_reward_counter = 0

    while True:        
        grayscaled_state = prep.convert_to_grayscale(state=state)
        states_queue.append(grayscaled_state)
        agent_state = prep.deque_to_tensor(states_queue)
        action = agent.select_action(agent_state, inference_only=True)

        state, reward, terminated, truncated, info = env.step(action=action)
        print(reward)

        if reward < 0:
            non_positive_reward_counter += 1
        else:
            non_positive_reward_counter = 0
        
        if non_positive_reward_counter >= 200:
            terminated = True

        if terminated or truncated:
            break

env.close()