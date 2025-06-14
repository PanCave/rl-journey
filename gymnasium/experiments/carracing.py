import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from collections import deque
import torch

from agents.carracing_agent import MichaelSchumacherDiscrete, DQN
from utils.dataclasses import Replay
import utils.preprocessing as prep
import utils.checkpoints as chkpts
import utils.batch_sampling as bts
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

BATCH_SIZE = 256
REPLAY_BUFFER_RESET_STEPS = 1000

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'  # GOAT
else:
    device = 'cpu'

# 0 nothing
# 1 left
# 2 right
# 3 gas
# 4 brake
env = gym.make('CarRacing-v3', render_mode='rgb_array', lap_complete_percent=0.95, domain_randomize=True, continuous=False, max_episode_steps=-1)

NUM_EPISODES = 10_000
NUM_TIMESTEPS = 10_000
MAX_REPLAY_BUFFER_LENGTH = 10_000
EPISODE_SAVE_RATE = 25
EXPERIMENT_NAME = 'master_lrschedule/'
CHECKPOINTS_PARENT_DIRECTORY = 'gymnasium/checkpoints/carracing_master/'
CHECKPOINTS_SAVE_SUB_DIRECTORY = EXPERIMENT_NAME
CHECKPOINTS_LOAD_SUB_DIRECTORY = 'master_nonlinear/'
CHECKPOINTS_SAVE_PATH = CHECKPOINTS_PARENT_DIRECTORY + CHECKPOINTS_SAVE_SUB_DIRECTORY + 'episode_{episode_idx}.pth'
REPEAT_ACTION_NUMBER = 6

replay_buffer_reset_step_counter = 0

writer = SummaryWriter("gymnasium/runs/carracing_master/" + EXPERIMENT_NAME)

checkpoint = None
LOAD_EPISODE = -1
load_checkpoint_path = CHECKPOINTS_PARENT_DIRECTORY + CHECKPOINTS_LOAD_SUB_DIRECTORY + f'episode_{LOAD_EPISODE}.pth'
if os.path.exists(load_checkpoint_path):
    checkpoint = chkpts.load_checkpoint(load_checkpoint_path=load_checkpoint_path)

state_width = 84
state_height = 84
number_of_frames = 4
input_shape = (state_width, state_height, number_of_frames)
output_shape = 5
dqn = DQN(input_shape=input_shape, action_dim=output_shape)
optimizer = Adam(dqn.parameters(), lr=0.001)
lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
agent = MichaelSchumacherDiscrete(
    env=env,
    num_target_update_steps=2000,
    epsilon_init=1,    # Startwert für Epsilon
    epsilon_min=0.01, # Minimaler Epsilon-Wert
    epsilon_decay_rate=0.995,      # Abnahmerate von Epsilon
    gamma=0.95,          # Discount-Faktor
    optimizer=optimizer,
    device=device,
    policy_network=dqn
)
empty_state = torch.zeros(state_width, state_height)
replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_LENGTH)
global_step_counter = 0
episode_start_number = 0

if checkpoint is not None:
    # TODO: Reactivate, once lr is properly set in checkpoint
    # agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    episode_start_number = checkpoint['episode_idx']


for episode_idx in range(episode_start_number, NUM_EPISODES):
    
    state, info = env.reset()

    non_positive_reward_counter = 0
    sum_episode_reward = 0
    sum_episode_loss = 0
    episode_step_counter = 0

    print(f'Episode {episode_idx}, Epsilon: {agent.epsilon}')
    writer.add_scalar("Epsilon", agent.epsilon, episode_idx)

    grayscaled = prep.convert_to_grayscale(state)
    states_queue = deque(maxlen=number_of_frames, iterable=[empty_state]*(number_of_frames - 1) + [grayscaled])

    for _ in range(50):
        env.step(0)
    
    for timestep in range(NUM_TIMESTEPS):
        global_step_counter += 1

        grayscaled_state = prep.convert_to_grayscale(state=state)
        states_queue.append(grayscaled_state)
        agent_state = prep.deque_to_tensor(states_queue)
        action = agent.select_action(agent_state)

        repeat_action_reward = 0
        for _ in range(REPEAT_ACTION_NUMBER):
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_step_counter += 1
            repeat_action_reward += reward

            next_grayscaled_state = prep.convert_to_grayscale(state=next_state)
            states_queue.append(next_grayscaled_state)

            if reward < 0:
                non_positive_reward_counter += 1
            else:
                non_positive_reward_counter = 0
            
            if non_positive_reward_counter >= 50 + (agent.epsilon * 150):
                print(f'Episode {episode_idx} abgebrochen nach {episode_step_counter} Schritten')
                terminated = True

            if truncated or terminated:
                break

            state = next_state
        

        next_agent_state = prep.deque_to_tensor(states_queue)

        sum_episode_reward += repeat_action_reward
        
        experience = Replay(agent_state, action, repeat_action_reward, next_agent_state, terminated or truncated)
        replay_buffer.append(experience)
        
        if len(replay_buffer) >= BATCH_SIZE and timestep % 4 == 0:
            batch = bts.sample_with_high_rewards_prioritized(replay_buffer=replay_buffer, number_of_samples=BATCH_SIZE)
            loss = agent.train(batch)
            sum_episode_loss += loss
        
        if terminated or truncated:
            break
        
        state = next_state

    agent.update_epsilon()
    lr_scheduler.step()

    mean_episode_reward = sum_episode_reward / episode_step_counter
    lr = lr_scheduler.get_last_lr()
    writer.add_scalar("Learning Rate", lr[-1], episode_idx)
    writer.add_scalar("Summed Reward per Episode", sum_episode_reward, episode_idx)
    writer.add_scalar("Mean Reward per Episode", mean_episode_reward, episode_idx)
    writer.add_scalar("Summed Loss per Episode", sum_episode_loss, episode_idx)
    writer.add_scalar("Episode Step Counter", episode_step_counter, episode_idx)

    if episode_idx > 0 and episode_idx % EPISODE_SAVE_RATE == 0:
        chkpts.save_checkpoint(
            agent=agent, 
            episode_idx=episode_idx, 
            save_checkpoint_path_str=CHECKPOINTS_SAVE_PATH.format(episode_idx=episode_idx)
        )

    if agent.epsilon <= agent.epsilon_init * 0.2:
        agent.epsilon_init *= 0.7
        agent.reset_epsilon()
    

env.close()