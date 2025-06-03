import random

from collections import deque
from typing import List

from utils.dataclasses import Replay

class ReplayBufferSampler():
    def sample_with_high_rewards_prioritized(self, replay_buffer: deque, number_of_samples: int) -> List[Replay]:
        sampled_replays = random.sample(replay_buffer, min(len(replay_buffer), number_of_samples * 2))
        # Only keep the samples with the highest reward
        sorted_replays = sorted(sampled_replays, key=lambda replay: replay.reward, reverse=True)
        return sorted_replays[:number_of_samples]