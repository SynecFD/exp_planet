from numpy import ndarray
from torch.utils.data.dataset import Dataset

from model.experience_replay import ExperienceReplay


class ReplayBufferSet(Dataset):

    def __init__(self, buffer: ExperienceReplay, seq_length: int) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        super().__init__()
        self.buffer = buffer
        self.seq_length = seq_length

    def __getitem__(self, index: tuple[int, int]) -> tuple[ndarray, ndarray, ndarray]:
        return self.buffer.get_sample(*index, self.seq_length)

    def __len__(self) -> int:
        return len(self.buffer)
