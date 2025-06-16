import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from collections import deque
import random
import torch

from agents.carracing_agent import MichaelSchumacherDiscrete, DQN
from utils.dataclasses import Replay
import utils.preprocessing as prep
import utils.checkpoints as chkpts

BATCH_SIZE = 32

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'  # GOAT
else:
    device = 'cpu'

checkpoint = None
LOAD_EPISODE = 930
load_checkpoint_path = f'gymnasium/checkpoints/carracing_master/episode_{LOAD_EPISODE}.pth'
if os.path.exists(load_checkpoint_path):
    checkpoint = chkpts.load_checkpoint(load_checkpoint_path=load_checkpoint_path)

# 0 nothing
# 1 left
# 2 right
# 3 gas
# 4 brake
env = gym.make('CarRacing-v3', render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=True, continuous=False)

NUM_EPISODES = 3_000
MAX_REPLAY_BUFFER_LENGTH = 30_000
EPISODE_SAVE_RATE = 10
TRAN_FREQUENCY = 4
CHECKPOINTS_PATH = 'gymnasium/checkpoints/carracing_master/episode_{episode_idx}'

replay_buffer_reset_step_counter = 0

write = SummaryWriter("gymnasium/runs/carracing_master")

state_width = 96
state_height = 96
number_of_frames = 4
input_shape = (state_width, state_height, number_of_frames)
output_shape = 5
dqn = DQN(input_shape=input_shape, action_dim=output_shape)
optimizer = torch.optim.Adam(dqn.parameters())
agent = MichaelSchumacherDiscrete(
    env=env,
    num_target_update_steps=2_000,
    epsilon_init=1,    # Startwert für Epsilon
    epsilon_min=0.001, # Minimaler Epsilon-Wert
    epsilon_decay_rate=0.9999925,      # Abnahmerate von Epsilon
    gamma=0.95,          # Discount-Faktor
    optimizer=optimizer,
    device=device,
    policy_network=dqn
)
empty_state = torch.zeros(state_width, state_height)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
states_queue = deque(maxlen=number_of_frames)

episode_start_number = 0
if checkpoint is not None:
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    episode_start_number = checkpoint['episode_idx']

for episode_idx in range(episode_start_number, NUM_EPISODES):    
    write.add_scalar("Epsilon / Episode", agent.epsilon, episode_idx)

    state, info = env.reset()
    non_positive_reward_counter = 0
    sum_episode_reward = 0
    sum_episode_loss = 0
    episode_step_counter = 0

    for _ in range(number_of_frames):
        next_state, _, _, _, _ = env.step(0)
        grayscaled_next_state = prep.convert_to_grayscale(next_state)
        states_queue.append(grayscaled_next_state)

    while True:
        agent_state = prep.deque_to_tensor(states_queue)
        action = agent.select_action(agent_state)
        
        total_reward = 0
        for _ in range(1):
            next_state, reward, terminated, truncated, info = env.step(action)
            grayscaled_next_state = prep.convert_to_grayscale(next_state)
            states_queue.append(grayscaled_next_state)
            episode_step_counter += 1
            total_reward += reward
            
            if truncated or terminated:
                break

        sum_episode_reward += total_reward

        next_agent_state = prep.deque_to_tensor(states_queue)
        
        experience = Replay(agent_state, action, total_reward, next_agent_state, terminated or truncated)
        replay_buffer.append(experience)
        
        experience_buffer = list(replay_buffer)
        if (episode_step_counter % TRAN_FREQUENCY == 0) and (len(experience_buffer) >= BATCH_SIZE):
            batch = random.sample(experience_buffer, BATCH_SIZE)
            loss = agent.train(batch)
            sum_episode_loss += loss

        if terminated or truncated:
            break
        

    write.add_scalar("Summed Reward / Episode", sum_episode_reward, episode_idx)
    write.add_scalar("Summed Loss / Episode", sum_episode_loss, episode_idx)

    write.add_scalar("Episode Step Counter", episode_step_counter, episode_idx)

    if episode_idx > 0 and episode_idx % EPISODE_SAVE_RATE == 0:
        chkpts.save_checkpoint(
            agent=agent, 
            episode_idx=episode_idx, 
            save_checkpoint_path=CHECKPOINTS_PATH.format(episode_idx=episode_idx)
        )
    
env.close()