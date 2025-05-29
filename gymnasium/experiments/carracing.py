import gymnasium as gym

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import deque
import random

from agents.carracing_agent import MichaelSchumacherDiscrete, DQN
from utils.dataclasses import Replay

BATCH_SIZE = 64
REPLAY_BUFFER_RESET_STEPS = 1000

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

input_shape = (96, 96, 3)
output_shape = 5
dqn = DQN(input_shape=input_shape, action_dim=output_shape)
agent = MichaelSchumacherDiscrete(
    env=env,
    num_target_update_steps=100,
    epsilon_init=1,    # Startwert für Epsilon
    epsilon_min=0.001, # Minimaler Epsilon-Wert
    epsilon_decay_rate=0.9999,      # Abnahmerate von Epsilon
    gamma=0.9,          # Discount-Faktor
    policy_network=dqn
)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
step_counter = 0
network_train_rate = 4

for episode in range(NUM_EPISODES):
    
    state, info = env.reset()
    agent.reset_epsilon()
    
    for _ in range(NUM_TIMESTEPS):
        # TODO: Construct from 3 images
        action = agent.select_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        experience = Replay(state, action, reward, next_state, terminated or truncated)
        replay_buffer.append(experience)        
        
        if terminated or truncated:
            state, info = env.reset()
            break

        if step_counter % network_train_rate == 0 and len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            agent.train(batch, global_step=step_counter)
        
        step_counter += 1
        state = next_state

env.close()