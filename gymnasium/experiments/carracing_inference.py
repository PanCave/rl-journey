import gymnasium as gym

import sys
import os

import torch
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import deque

from agents.carracing_agent import MichaelSchumacherDiscrete, DQN
from utils.preprocessing import PreProcessor
import numpy as np

env = gym.make('CarRacing-v3', render_mode='human', lap_complete_percent=0.95, domain_randomize=True, continuous=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
preprocessor = PreProcessor(device)

load_episode = 1030
checkpoint_path = f"gymnasium/checkpoints/policy_best.pth"
checkpoint = None
if os.path.exists(checkpoint_path):
    print(f'Lade checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)


state_width = 96
state_height = 96
number_of_frames = 3
input_shape = (state_width, state_height, number_of_frames)
output_shape = 5
dqn = DQN(input_shape=input_shape, action_dim=output_shape)
agent = MichaelSchumacherDiscrete(
    env=env,
    epsilon_init=0.64,    # Startwert für Epsilon
    epsilon_min=0.001, # Minimaler Epsilon-Wert
    epsilon_decay_rate=0.999,      # Abnahmerate von Epsilon
    gamma=0.9,          # Discount-Faktor
    policy_network=dqn,
    optimizer=torch.optim.Adam(dqn.parameters(), lr=0.0003),
    summary_writer=None,
    device=device
)


states_queue = deque(maxlen=number_of_frames)
empty_state = np.zeros(input_shape, dtype=np.uint8)
for _ in range(number_of_frames):
    states_queue.append(empty_state)

for _ in range(10):
    state, info = env.reset()

    while True:
        states_queue.append(state)
        grayscaled_tensor = preprocessor.process(states_deque=states_queue)
        action = agent.select_action(grayscaled_tensor, inference_only=True)
        
        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state

        if terminated or truncated:
            break

env.close()