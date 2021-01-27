import numpy as np
from collections import deque, namedtuple
from typing import Tuple
import torch

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done'])


class ExperienceReplay:

    def __init__(self, size: int) -> None:
        self.replay = deque(maxlen=size)
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.replay)

    def append(self, experience: Experience) -> None:
        self.replay.append(experience)

    def sample(self, batch_size: int, length: int) -> Tuple[np.array, np.array, np.array, np.array]:
        terminal_states = np.where(getattr(self.replay, 'done'))[0]
        indices = np.random.choice(len(self.replay), batch_size, replace=False)
        states, actions, rewards, dones = [], [], [], []
        for _ in range(batch_size):
            invalid_idx = True
            while invalid_idx:
                idx = self.rng.integers(low=0, high=len(self.replay) - length)
                final_idx = idx + length - 1
                closest_terminal = np.where(terminal_states >= idx).min()
                invalid_idx = final_idx > closest_terminal
            states.append(getattr(self.replay, 'state')[idx])
            actions.append(getattr(self.replay, 'action')[idx])
            rewards.append(getattr(self.replay, 'reward')[idx])
            dones.append(getattr(self.replay, 'done')[idx])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool))

