from collections import deque, namedtuple
from functools import partial
from itertools import chain, compress
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import RandomSampler
from torch.utils.data._utils import collate

from util import pad_sequence

Experience = namedtuple('Experience', field_names=['states', 'actions', 'rewards'])


class ExperienceReplay:

    def __init__(self, size: int) -> None:
        self.replay = deque(maxlen=size)
        self.current_frame_buffer = []
        self.current_action_buffer = []
        self.current_reward_buffer = []

    def __len__(self) -> int:
        return len(self.replay)

    def append(self, experience: Experience) -> None:
        self.replay.append(experience)

    def add_step_data(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.current_frame_buffer.append(state)
        self.current_action_buffer.append(action)
        self.current_reward_buffer.append(reward)

    def stack_episode(self) -> None:
        if not self.current_frame_buffer and not self.current_action_buffer and not self.current_reward_buffer:
            return
        states = np.stack(self.current_frame_buffer)
        actions = np.stack(self.current_action_buffer)
        rewards = np.array(self.current_reward_buffer)
        self.append(Experience(states, actions, rewards))
        # reset buffer for new episode
        self.current_frame_buffer.clear()
        self.current_action_buffer.clear()
        self.current_reward_buffer.clear()

    def get_sample(self, idx: int, seq_start: int, length: int) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        episode = self.replay[idx]
        seq_end = min(episode.states.shape[0], seq_start + length)

        states = episode.states[seq_start:seq_end]
        actions = episode.actions[seq_start:seq_end]
        rewards = episode.rewards[seq_start:seq_end]

        return states, actions, rewards

    def persist(self, path: Path) -> None:
        if path.suffix is not None and path.suffix != ".npz":
            raise ValueError("ExperienceReplay save-path must end on .npz or have no file extension")
        path.parent.mkdir(exist_ok=True)
        episodes, actions, rewards = list(zip(*self.replay))
        np.savez_compressed(path, episodes=episodes, actions=actions, rewards=rewards)

    def load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError("Invalid path to persisted ExperienceReplay")
        dict = np.load(path, allow_pickle=False, fix_imports=False)
        states = dict["episodes"], dict["actions"], dict["rewards"]
        exp_rpl = list(map(lambda state: Experience(*state), zip(*states)))
        self.replay.extend(exp_rpl)


class ExperienceReplaySampler(RandomSampler):

    def __init__(self, buffer: Sequence, seq_length: int, allow_padding: bool = True, generator=None) -> None:
        super().__init__(data_source=buffer, generator=generator)
        self.seq_length = seq_length
        self.allow_padding = allow_padding

    def __iter__(self) -> tuple[int, int]:
        for idx in super().__iter__():
            high = self.data_source[idx].states.shape[0]
            if not self.allow_padding:
                high -= self.seq_length
            seq_start = torch.randint(high=high, size=(1,), dtype=torch.int64,
                                      generator=self.generator).item()
            yield idx, seq_start


def experience_replay_collate(batch: list[tuple[Tensor, Tensor, Tensor]]):
    conversion = collate.default_convert(batch)
    pad_sequence_no_sort = partial(pad_sequence, enforce_sorted=False)
    padded_batch = chain(*map(pad_sequence_no_sort, zip(*conversion)))
    return list(compress(padded_batch, [1, 0, 1, 0, 1, 1]))
