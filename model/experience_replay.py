from collections import deque, namedtuple
from functools import partial
from itertools import chain, compress
from pathlib import Path
from typing import Sequence, Optional

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

    def add_step_data(self, state: Tensor, action: np.ndarray, reward: float) -> None:
        self.current_frame_buffer.append(state)
        self.current_action_buffer.append(action)
        self.current_reward_buffer.append(reward)

    def stack_episode(self) -> None:
        if len(self.current_frame_buffer) > 1 and len(self.current_action_buffer) > 1 \
                and len(self.current_reward_buffer) > 1:
            states = torch.stack(self.current_frame_buffer)
            actions = np.stack(self.current_action_buffer)
            rewards = np.array(self.current_reward_buffer)
            self.append(Experience(states, actions, rewards))
        # reset buffer for new episode
        self.current_frame_buffer.clear()
        self.current_action_buffer.clear()
        self.current_reward_buffer.clear()

    def get_sample(self, idx: int, seq_start: int, length: int) -> tuple[Tensor, np.ndarray, np.ndarray]:
        episode = self.replay[idx]
        seq_end = min(episode.states.size(0), seq_start + length)

        states = episode.states.narrow(0, seq_start, min(length, episode.states.size(0) - seq_start))
        actions = episode.actions[seq_start:seq_end]
        rewards = episode.rewards[seq_start:seq_end]

        return states, actions, rewards

    def persist(self, path: Path) -> None:

        path.parent.mkdir(exist_ok=True)
        episodes, actions, rewards = list(zip(*self.replay))
        np.savez_compressed(path.with_suffix(".npz"), actions=actions, rewards=rewards)
        torch.save(episodes, path.with_suffix(".pt"))

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError("Invalid path to persisted ExperienceReplay")
        dict = np.load(path.with_suffix(".npz"), allow_pickle=False, fix_imports=False)
        states = torch.load(path.with_suffix(".pt")), dict["actions"], dict["rewards"]
        exp_rpl = list(map(lambda state: Experience(*state), zip(*states)))
        self.replay.extend(exp_rpl)


class StubExperienceReplay(ExperienceReplay):
    def __init__(self) -> None:
        super().__init__(0)

    def __len__(self) -> int:
        return 0

    def append(self, experience: Experience) -> None:
        pass

    def add_step_data(self, state: Tensor, action: np.ndarray, reward: float) -> None:
        pass

    def stack_episode(self) -> None:
        pass

    def get_sample(self, idx: int, seq_start: int, length: int) -> tuple[Tensor, np.ndarray, np.ndarray]:
        pass

    def persist(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class ExperienceReplaySampler(RandomSampler):

    def __init__(self, buffer: Sequence, seq_length: int, allow_padding: bool = True, replacement: bool = False,
                 num_samples: Optional[int] = None, generator: Optional[torch.Generator] = None) -> None:
        super().__init__(data_source=buffer, replacement=replacement, num_samples=num_samples, generator=generator)
        self.seq_length = seq_length
        self.allow_padding = allow_padding

    def __iter__(self) -> tuple[int, int]:
        for idx in super().__iter__():
            high = self.data_source[idx].states.shape[0]
            if not self.allow_padding:
                high -= self.seq_length
            seq_start = torch.randint(high=high, size=(1,), dtype=torch.int64, generator=self.generator).item()
            yield idx, seq_start


def experience_replay_collate(batch: list[tuple[Tensor, Tensor, Tensor]]) -> list[Tensor, Tensor, Tensor, Tensor]:
    conversion = collate.default_convert(batch)
    pad_sequence_no_sort = partial(pad_sequence, enforce_sorted=False)
    padded_batch = chain(*map(pad_sequence_no_sort, zip(*conversion)))
    obs, actions, rewards, length = compress(padded_batch, [1, 0, 1, 0, 1, 1])
    return [obs, actions.float(), rewards.float(), length]
