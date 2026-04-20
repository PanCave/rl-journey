import random

from collections import deque
from typing import List

from utils.dataclasses import Replay, ReplayContinuous

def sample_with_high_rewards_prioritized(replay_buffer: deque, number_of_samples: int) -> List[Replay]:
    sampled_replays = random.sample(replay_buffer, min(len(replay_buffer), number_of_samples * 2))
    # Only keep the samples with the highest reward
    sorted_replays = sorted(sampled_replays, key=lambda replay: replay.reward, reverse=True)
    return sorted_replays[:number_of_samples]

def sample_continuous_with_high_rewards_prioritized(replay_buffer: deque, number_of_samples: int) -> List[ReplayContinuous]:
    sampled_replays = random.sample(replay_buffer, min(len(replay_buffer), number_of_samples * 2))
    # Only keep the samples with the highest reward
    sorted_replays = sorted(sampled_replays, key=lambda replay: replay.reward, reverse=True)
    return sorted_replays[:number_of_samples]

def sample_continuous(replay_buffer: deque, number_of_samples: int) -> List[ReplayContinuous]:
    sampled_replays = random.sample(replay_buffer, min(len(replay_buffer), number_of_samples))
    return sampled_replays
