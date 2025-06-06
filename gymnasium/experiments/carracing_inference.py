import sys
import os
from collections import deque

import gymnasium as gym
import torch

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.checkpoints as chkpts
from agents.carracing_agent import DQN, MichaelSchumacherDiscrete
import utils.preprocessing as prep

LOAD_EPISODE = 1425
CHECKPOINT_PATH = f'gymnasium/checkpoints/carracing_master/master_lowerlr_nomaxsteps/episode_{LOAD_EPISODE}.pth'
checkpoint = chkpts.load_checkpoint(CHECKPOINT_PATH)

env = gym.make('CarRacing-v3', render_mode='human', lap_complete_percent=0.95, domain_randomize=True, continuous=False, max_episode_steps=-1)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'  # GOAT
else:
    device = 'cpu'

state_width = 96
state_height = 96
number_of_frames = 3
input_shape = (state_width, state_height, number_of_frames)
output_shape = 5
dqn = DQN(input_shape=input_shape, action_dim=output_shape)
optimizer = torch.optim.Adam(dqn.parameters())
agent = MichaelSchumacherDiscrete(
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

empty_state = torch.zeros(state_width, state_height)
states_queue = deque(maxlen=number_of_frames, iterable=[empty_state] * 3)

agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])

for _ in range(20):
    state, _ = env.reset()
    non_positive_reward_counter = 0

    while True:
        grayscaled_state = prep.convert_to_grayscale(state=state)
        states_queue.append(grayscaled_state)
        agent_state = prep.deque_to_tensor(states_queue)
        action = agent.select_action(agent_state, inference_only=True)

        state, reward, terminated, truncated, info = env.step(action=action)

        if reward < 0:
            non_positive_reward_counter += 1
        else:
            non_positive_reward_counter = 0
        
        if non_positive_reward_counter >= 200:
            terminated = True
        
        if info != {}:
            print(info)

        if terminated or truncated:
            break

env.close()