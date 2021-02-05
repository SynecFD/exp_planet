from torch import Tensor, rand_like, float32, cat, split
from typing import Tuple, Union, List


def preprocess_observation_(observation: Tensor, bit_depth: int = 5) -> Tensor:
    """Reduction from float32-Tensor [0, 255] to [-0.5, 0.5]

    Reduces to the given bit depth and centers to [-0.5, 0.5]
    In addition, adds uniform random noise to the image
    """
    assert observation.shape[-3:] == (64, 64, 3), 'Input obs of wrong shape'
    observation = observation.to(dtype=float32, copy=True)
    observation.floor_divide_(2 ** (8 - bit_depth)).div_(2 ** bit_depth).sub_(0.5)
    observation.add_(rand_like(observation).div_(2 ** bit_depth))
    h, w, c = range(observation.ndim - 3, observation.ndim)
    observation = observation.permute(c, h, w)
    return observation


def concatenate_batch_sequences(batch_seq: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tuple[Tensor, List[int]]:
    seq_lengths = [seq.shape[0] for seq in batch_seq]
    concat = cat(batch_seq)
    return concat, seq_lengths


def split_into_batch_sequences(tensor: Tensor, seq_lengths: List[int]) -> Tuple[Tensor, ...]:
    return split(tensor, seq_lengths)
