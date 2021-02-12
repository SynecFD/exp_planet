import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from model.experience_replay import ExperienceReplay
from typing import Tuple


class ReplayLoader(nn.Module):

    def __init__(self, replay_size: int, episode_length: int, batch_size: int) -> None:
        self.buffer = ExperienceReplay(replay_size)
        self.episode_length = episode_length
        self.batch_size = batch_size

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ReplayBufferSet(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=None,
        )
        return dataloader

    def get_dataloader(self) -> DataLoader:
        return self.__dataloader()


class ReplayBufferSet(IterableDataset):
    def __init__(self, buffer: ExperienceReplay, sample_size: int) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i]
