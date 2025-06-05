import gymnasium as gym
import numpy as np

import time

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import deque
import random
import torch

from agents.carracing_agent import MichaelSchumacherDiscrete, DQN
from utils.dataclasses import Replay
from utils.preprocessing import Preprocessor

BATCH_SIZE = 64
REPLAY_BUFFER_RESET_STEPS = 1000

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# 0 nothing
# 1 left
# 2 right
# 3 gas
# 4 brake
env = gym.make('CarRacing-v3', render_mode='human', lap_complete_percent=0.95, domain_randomize=True, continuous=False)

NUM_EPISODES = 100
NUM_TIMESTEPS = 1000
MAX_REPLAY_BUFFER_LENGTH = 10_000

replay_buffer_reset_step_counter = 0

preprocessor = Preprocessor(device=device)

state_width = 96
state_height = 96
number_of_frames = 3
input_shape = (state_width, state_height, number_of_frames)
output_shape = 5
dqn = DQN(input_shape=input_shape, action_dim=output_shape)
agent = MichaelSchumacherDiscrete(
    env=env,
    num_target_update_steps=100,
    epsilon_init=1,    # Startwert für Epsilon
    epsilon_min=0.001, # Minimaler Epsilon-Wert
    epsilon_decay_rate=0.995,      # Abnahmerate von Epsilon
    gamma=0.9,          # Discount-Faktor
    device=device,
    policy_network=dqn
)
empty_state = torch.zeros(state_width, state_height)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
states_queue = deque(maxlen=number_of_frames, iterable=[empty_state] * 3)
next_states_queue = deque(maxlen=number_of_frames, iterable=[empty_state] * 3)

for episode in range(NUM_EPISODES):
    
    state, info = env.reset()
    agent.reset_epsilon()

    for _ in range(50):
        # Fill states queues
        env.step(0)
    
    for _ in range(NUM_TIMESTEPS):
        # TODO: Construct from 3 images
        grayscaled_state = preprocessor.convert_to_grayscale(state=state)
        states_queue.append(grayscaled_state)
        action = agent.select_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        grayscaled_next_state = preprocessor.convert_to_grayscale(next_state)
        next_states_queue.append(grayscaled_next_state)
        
        experience = Replay(torch.stack(list(states_queue)), action, reward, torch.stack(list(states_queue)), terminated or truncated)
        replay_buffer.append(experience)
        
        experience_buffer = list(replay_buffer)
        if len(experience_buffer) >= BATCH_SIZE:
            batch = random.sample(experience_buffer, BATCH_SIZE)
            agent.train(batch)        
        
        if terminated or truncated:
            state, info = env.reset()
            break
        
        state = next_state  

env.close()