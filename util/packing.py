from typing import Tuple, Union, List

import torch
from more_itertools import all_equal, is_sorted, sort_together
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def concatenate_batch_sequences(batch_seq: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tuple[Tensor, List[int]]:
    seq_lengths = [seq.size(0) for seq in batch_seq]
    concat = torch.cat(batch_seq)
    return concat, seq_lengths


def split_into_batch_sequences(tensor: Tensor, seq_lengths: List[int]) -> Tuple[Tensor, ...]:
    return torch.split(tensor, seq_lengths)


def pack_padded_sequence(batch: Tensor, lengths: List[int], enforce_sorted: bool = True) -> PackedSequence:
    if enforce_sorted and not is_sorted(lengths, reverse=True):
        rows = list(range(batch.size(0)))
        # https://stackoverflow.com/a/45514542/4907774
        lengths, rows = map(list, sort_together([lengths, rows], reverse=True))
        batch = batch[rows, ...]
    return torch.nn.utils.rnn.pack_padded_sequence(batch, lengths, batch_first=True, enforce_sorted=enforce_sorted)


def pad_sequence(batch_seq: Union[Tuple[Tensor, ...], List[Tensor]], enforce_sorted: bool = True) \
        -> Tuple[Tensor, List[int]]:
    lengths = [seq.size(0) for seq in batch_seq]
    if not all_equal(lengths):
        # Sorting due to ``enforce_sorted = True`` default flag, for potential ONNX export.
        if enforce_sorted and not is_sorted(lengths, reverse=True):
            # https://stackoverflow.com/a/45514542/4907774
            lengths, batch_seq = map(list, sort_together([lengths, batch_seq], reverse=True))
        return torch.nn.utils.rnn.pad_sequence(batch_seq, batch_first=True), lengths
    else:
        # Batch-first
        return torch.stack(batch_seq), lengths
