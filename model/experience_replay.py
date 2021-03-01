from collections import deque, namedtuple
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import RandomSampler

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
        states = np.stack(self.current_frame_buffer)
        actions = np.stack(self.current_action_buffer)
        rewards = np.array(self.current_reward_buffer)
        self.append(Experience(states, actions, rewards))
        # reset buffer for new episode
        self.current_frame_buffer.clear()
        self.current_action_buffer.clear()
        self.current_reward_buffer.clear()

    def get_sample(self, idx: int, seq_start: int, length: int) \
            -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        state_list, action_list, reward_list = [], [], []
        episode = self.replay[idx]
        assert seq_start + length <= episode.shape[0]

        states = episode.states[seq_start:(length + seq_start)]
        actions = episode.actions[seq_start:(length + seq_start)]
        rewards = episode.rewards[seq_start:(length + seq_start)]
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)

        return state_list, action_list, reward_list

    def persist(self, path: Path) -> None:
        if path.suffix is not None and path.suffix != ".npz":
            raise ValueError("ExperienceReplay save-path must end on .npz or have no file extension")
        path.parent.mkdir(exist_ok=True)
        episodes, actions, rewards = list(zip(*self.replay))
        np.savez_compressed(path, episode=episodes, actions=actions, rewards=rewards)

    def load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError("Invalid path to persisted ExperienceReplay")
        dict = np.load(path, allow_pickle=False, fix_imports=False)
        states = dict["episodes"], dict["actions"], dict["rewards"]
        exp_rpl = list(map(lambda state: Experience(*state), zip(*states)))
        self.replay.extend(exp_rpl)


class ExperienceReplaySampler(RandomSampler):

    def __init__(self, buffer: Sequence, seq_length: int, generator=None) -> None:
        super().__init__(data_source=buffer, generator=generator)
        self.seq_length = seq_length

    def __iter__(self) -> tuple[int, int]:
        for idx in super().__iter__():
            max_length = min(self.seq_length, self.data_source[idx].states.shape[0])
            seq_start = torch.randint(high=max_length - 1, size=(1,), dtype=torch.int64,
                                      generator=self.generator).item()
            yield idx, seq_start
