import gymnasium as gym

import sys
import os

import torch
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from agents.carracing_agent import MichaelSchumacherDiscrete, DQN
from utils.dataclasses import Replay
from utils.preprocessing import PreProcessor
from utils.batch_sampling import ReplayBufferSampler
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

BATCH_SIZE = 256

# 0 nothing
# 1 left
# 2 right
# 3 gas
# 4 brake
env = gym.make('CarRacing-v3', render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=True, continuous=False)

writer = SummaryWriter(log_dir="gymnasium/runs/carracing_experiment")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
preprocessor = PreProcessor(device)
batch_sampler = ReplayBufferSampler()

load_episode = 840
checkpoint_path = f"gymnasium/checkpoints/policy_full_ep{load_episode}.pth"
checkpoint = None
if os.path.exists(checkpoint_path):
    print(f'Lade checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)

NUM_EPISODES = 10_000
NUM_TIMESTEPS = 1_000
MAX_REPLAY_BUFFER_LENGTH = 50_000

replay_buffer_reset_step_counter = 0

state_width = 96
state_height = 96
number_of_frames = 3
input_shape = (state_width, state_height, number_of_frames)
output_shape = 5
dqn = DQN(input_shape=input_shape, action_dim=output_shape)
optimizer = torch.optim.Adam(dqn.parameters(), lr=0.0003)
agent = MichaelSchumacherDiscrete(
    env=env,
    epsilon_init=1,    # Startwert für Epsilon
    epsilon_min=0.001, # Minimaler Epsilon-Wert
    epsilon_decay_rate=0.999,      # Abnahmerate von Epsilon
    gamma=0.9,          # Discount-Faktor
    policy_network=dqn,
    optimizer=optimizer,
    summary_writer=writer,
    device=device
)
learning_rate_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.95)

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
empty_state = torch.zeros(state_width, state_height, number_of_frames, dtype=torch.uint8)
states_queue = deque(maxlen=number_of_frames)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
step_counter = start_step_counter
epsilon_update_rate = 1
network_train_rate = 4
checkpoint_save_rate = 10
reward_history = []
moving_reward_window = 10
best_average_reward = float('-inf')
num_target_update_steps=500

for _ in range(number_of_frames):
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
            batch = batch_sampler.sample_with_high_rewards_prioritized(replay_buffer, BATCH_SIZE)
            agent.train(batch, global_step=step_counter)
            learning_rate_scheduler.step()

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

    if episode_idx % 100 == 0:
        with torch.no_grad():
            test_state = grayscaled_states_tensor.unsqueeze(0)
            q_values = agent.policy_network(test_state)
            print(f"Q-Values: {q_values.cpu().numpy()}")
            print("Actions: 0=Nothing, 1=Left, 2=Right, 3=Gas, 4=Brake")
        

env.close()