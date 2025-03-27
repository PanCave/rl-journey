import gymnasium as gym
import numpy as np

from collections import deque
import random

env = gym.make('CarRacing-v3', render_mode='human', lap_complete_percent=0.95, domain_randomize=True, continuous=True)

episodes = 100

state, info = env.reset()
for episode in range(episodes):
    replay_buffer = deque(maxlen=1000)
    
    for _ in range(500):
        action = env.action_space.sample()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        experience = (state, action, reward, next_state)
        replay_buffer.append(experience)
        
        if terminated or truncated:
            state, info = env.reset()
            break
    
    experience_buffer = list(replay_buffer)
    random.sample(experience_buffer)

env.close()