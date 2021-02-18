import numpy as np
import torch
from collections import deque, namedtuple

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

    def add_step_data(self, state: torch.tensor, action: torch.tensor, reward: float) -> None:
        self.current_frame_buffer.append(state)
        self.current_action_buffer.append(action)
        self.current_reward_buffer.append(reward)

    def stack_episode(self) -> None:
        states = torch.stack(self.current_frame_buffer)
        actions = torch.stack(self.current_action_buffer)
        rewards = torch.as_tensor(self.current_reward_buffer)
        self.append(Experience(states, actions, rewards))
        # reset buffer for new episode
        self.current_frame_buffer = []
        self.current_action_buffer = []
        self.current_reward_buffer = []

    def sample(self, batch_size: int, length: int) -> list[torch.tensor, torch.tensor, torch.tensor]:
        state_list, action_list, reward_list = [], [], []
        for _ in range(batch_size):
            episode_idx = self.rng.integers(low=0, high=len(self.replay))
            episode = self.replay[episode_idx]
            states = getattr(episode, 'states')
            actions = getattr(episode, 'actions')
            rewards = getattr(episode, 'rewards')
            starting_idx = -1
            while starting_idx < 0 or starting_idx > len(episode):
                starting_idx = self.rng.integers(low=0, high=length-1)

            states.narrow(0, starting_idx, length + starting_idx)
            actions.narrow(0, starting_idx, length + starting_idx)
            rewards.narrow(0, starting_idx, length + starting_idx)
            state_list.append(states)
            action_list.append(actions)
            reward_list.append(rewards)

        return state_list, action_list, reward_list

