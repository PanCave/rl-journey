import sys
import os
from collections import deque

import gymnasium as gym
import torch
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.checkpoints as chkpts
from agents.continuous_agent import SACAgent
import utils.preprocessing as prep
from networks.continuous_car_racing_cnn import ContinuousCarRacingPolicy

# Create gifs directory if it doesn't exist
VIDEO_DIRECTORY = 'gymnasium/videos/'
os.makedirs(VIDEO_DIRECTORY, exist_ok=True)

LOAD_EPISODE = -1
CHECKPOINTS_DIRECTORY = 'gymnasium/checkpoints/carracing_sac/'
EXPERIMENT_NAME = 'sac_cpu'
CHECKPOINT_PATH = CHECKPOINTS_DIRECTORY + EXPERIMENT_NAME + f'/episode_{LOAD_EPISODE}.pth'
checkpoint = chkpts.load_checkpoint(CHECKPOINT_PATH)

STATE_SLICES = (slice(6, -6), slice(None, -12), slice(None, None))

env = gym.make('CarRacing-v3', render_mode='human', lap_complete_percent=0.95, domain_randomize=True, continuous=True, max_episode_steps=-1)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'  # SCHMUTZ
else:
    #device = 'cpu'
    device = torch.device('cpu')

state_width = 84
state_height = 84
number_of_frames = 4
input_shape = (state_width, state_height, number_of_frames)
output_shape = 5
sac_policy = ContinuousCarRacingPolicy(input_shape=input_shape, action_dim=output_shape)
optimizer = torch.optim.Adam(sac_policy.parameters())
agent = SACAgent(
    env=env,
    policy_network=sac_policy,
    critic_1_network=None,
    critic_2_network=None,
    alpha=1,
    tau=0.95,
    gamma=0.995,
    critic_1_optimizer=None,
    critic_2_optimizer=None,
    policy_optimizer=None,
    device=device,
    action_range_mins=np.array([-1, 0, 0]),
    action_range_maxs=np.array([1, 1, 1])
)

empty_state = torch.zeros(state_width, state_height)
states_queue = deque(maxlen=number_of_frames, iterable=[empty_state] * 3)

if checkpoint:
    agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
# else:
#     raise ValueError("Checkpoint must not be None")

for episode_idx in range(20):
    state, _ = env.reset()
    non_positive_reward_counter = 0

    while True:        
        grayscaled_state = prep.convert_to_grayscale(state=state, slices=STATE_SLICES)
        states_queue.append(grayscaled_state)
        agent_state = prep.deque_to_tensor(states_queue)
        action = np.array([0, 1, 0])#agent.select_action(agent_state, inference_only=True)

        state, reward, terminated, truncated, info = env.step(action=action)

        if float(reward) < 0:
            non_positive_reward_counter += 1
        else:
            non_positive_reward_counter = 0
        
        if non_positive_reward_counter >= 200:
            terminated = True
        
        if info != {}:
            print(info)
            if info['lap_finished']:
                print(episode_idx)

        if terminated or truncated:
            break

env.close()