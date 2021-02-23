from collections import deque, namedtuple

import numpy as np
import torch

Experience = namedtuple('Experience', field_names=['states', 'actions', 'rewards'])


class ExperienceReplay:

    def __init__(self, size: int) -> None:
        self.replay = deque(maxlen=size)
        self.rng = np.random.default_rng()
        self.current_frame_buffer = []
        self.current_action_buffer = []
        self.current_reward_buffer = []

    def __len__(self) -> int:
        return len(self.replay)

    def append(self, experience: Experience) -> None:
        self.replay.append(experience)

    def add_step_data(self, state: torch.Tensor, action: torch.Tensor, reward: float) -> None:
        self.current_frame_buffer.append(state)
        self.current_action_buffer.append(action)
        self.current_reward_buffer.append(reward)

    def stack_episode(self) -> None:
        states = torch.stack(self.current_frame_buffer)
        actions = torch.stack(self.current_action_buffer)
        rewards = torch.as_tensor(self.current_reward_buffer)
        self.append(Experience(states, actions, rewards))
        # reset buffer for new episode
        self.current_frame_buffer.clear()
        self.current_action_buffer.clear()
        self.current_reward_buffer.clear()

    def sample(self, batch_size: int, length: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        episode_idx = self.rng.choice(len(self), size=batch_size, replace=False, shuffle=False)
        episodes = [self.replay[idx] for idx in episode_idx]
        starting_idx = self.rng.integers(low=0, high=[min(length, state.size(0)) - 1 for state, _, _ in episodes])
        # noinspection PyTypeChecker
        return tuple([ith_attrib.narrow(0, idx, length + idx) for ith_attrib, idx in zip(attr, starting_idx)]
                     for attr in zip(*episodes))
