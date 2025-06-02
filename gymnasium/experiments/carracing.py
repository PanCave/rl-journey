import gymnasium as gym

import sys
import os

import torch
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

from agents.carracing_agent import MichaelSchumacherDiscrete, DQN
from utils.dataclasses import Replay
from utils.preprocessing import PreProcessor
import numpy as np

def save_checkpoint(
        agent: DQN,
        episode_idx: int,
        step_counter: int,
        checkpoint_name: str
):
    save_dict = {
            'policy_network_state_dict': agent.policy_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode_idx': episode_idx,
            'epsilon': agent.epsilon,
            'step_counter': step_counter
        }
    torch.save(save_dict, f"gymnasium/checkpoints/{checkpoint_name}.pth")

BATCH_SIZE = 64
REPLAY_BUFFER_RESET_STEPS = 1000

# 0 nothing
# 1 left
# 2 right
# 3 gas
# 4 brake
env = gym.make('CarRacing-v3', render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=True, continuous=False)

writer = SummaryWriter(log_dir="gymnasium/runs/carracing_experiment")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessor = PreProcessor(device)

load_episode = 840
checkpoint_path = f"gymnasium/checkpoints/stuck_detection/policy_full_ep{load_episode}.pth"
checkpoint = None
if os.path.exists(checkpoint_path):
    print(f'Lade checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)

NUM_EPISODES = 1000
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
    epsilon_decay_rate=0.995,      # Abnahmerate von Epsilon
    gamma=0.9,          # Discount-Faktor
    policy_network=dqn,
    optimizer=torch.optim.Adam(dqn.parameters(), lr=0.0003),
    summary_writer=writer,
    device=device
)

start_step_counter = 0
start_episode_number = 0

if checkpoint is not None:
    agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
    agent.policy_network.eval()
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.target_network.eval()
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    start_episode_number = checkpoint['episode_idx']
    start_step_counter = checkpoint['step_counter']
empty_state = np.zeros((96, 96, 3), dtype=np.uint8)
states_queue = deque(maxlen=3)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
step_counter = start_step_counter
epsilon_update_rate = 1
network_train_rate = 4
checkpoint_save_rate = 10
reward_history = []
moving_reward_window = 10
best_average_reward = float('-inf')

states_queue.append(empty_state)
states_queue.append(empty_state)
states_queue.append(empty_state)



for episode_idx in range(start_episode_number, NUM_EPISODES):
    state, info = env.reset()
    cummulative_episode_reward = 0
    episode_timesteps = 0
    print(episode_idx)
    
    no_improvement_steps = 0
    max_no_improvement_steps = int(50 + agent.epsilon * 150)
    last_reward = cummulative_episode_reward
    early_stop = False
    
    for timestep in range(NUM_TIMESTEPS):
        episode_timesteps += 1
        states_queue.append(state)
        grayscaled_tensor = preprocessor.process(states_deque=states_queue)
        action = agent.select_action(grayscaled_tensor)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        cummulative_episode_reward += reward

        step_counter += 1

        if cummulative_episode_reward > last_reward:
            no_improvement_steps = 0
            last_reward = cummulative_episode_reward
        else:
            no_improvement_steps += 1

        if no_improvement_steps >= max_no_improvement_steps:
            early_stop = True
            writer.add_scalar("Vorzeitig abgebrochen bei Schritt", timestep, episode_idx)
            print(f'Episode {episode_idx} nach {timestep} Zeitschritte abgebrochen.')
            terminated = True
        
        experience = Replay(state, action, reward, next_state, terminated or truncated)
        replay_buffer.append(experience)        
        
        if step_counter % network_train_rate == 0 and len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            agent.train(batch, global_step=step_counter)

        if step_counter % num_target_update_steps == 0:
            agent.update_target_network()
        
        state = next_state

        if terminated or truncated:
            break
    
    if not early_stop:
        writer.add_scalar("Vorzeitig abgebrochen bei Schritt", NUM_TIMESTEPS, episode_idx)

    if episode_idx % epsilon_update_rate == 0:
        agent.update_epsilon()

    mean_episode_reward = cummulative_episode_reward / episode_timesteps
    writer.add_scalar("Mean Reward / Episode", mean_episode_reward, episode_idx)
    writer.add_scalar("Epsilon", agent.epsilon, episode_idx)

    reward_history.append(mean_episode_reward)

    if len(reward_history) >= moving_reward_window:
        moving_reward_average = sum(reward_history[-moving_reward_window:])/moving_reward_window
        if moving_reward_average > best_average_reward:
            best_average_reward = moving_reward_average
            save_checkpoint(agent=agent, episode_idx=episode_idx, step_counter=step_counter, checkpoint_name='policy_best')
        writer.add_scalar("Best Average Reward", best_average_reward, episode_idx)

    if episode_idx > 0 and episode_idx % checkpoint_save_rate == 0:
        save_checkpoint(agent=agent, episode_idx=episode_idx, step_counter=step_counter, checkpoint_name=f'policy_full_ep{episode_idx}')
        

env.close()