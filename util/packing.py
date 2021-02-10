from typing import Union

import torch
from more_itertools import all_equal, is_sorted, sort_together
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def concatenate_batch_sequences(batch_seq: Union[tuple[Tensor, ...], list[Tensor]]) -> tuple[Tensor, list[int]]:
    seq_lengths = [seq.size(0) for seq in batch_seq]
    concat = torch.cat(batch_seq)
    return concat, torch.as_tensor(seq_lengths, device=concat.device)


def split_into_batch_sequences(tensor: Tensor, seq_lengths: list[int]) -> tuple[Tensor, ...]:
    return torch.split(tensor, seq_lengths)


def pack_padded_sequence(batch: Tensor, lengths: Tensor, enforce_sorted: bool = True) -> PackedSequence:
    if enforce_sorted and not is_sorted(lengths, reverse=True):
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(batch.device)
        batch = batch.index_select(0, sorted_indices)
    return torch.nn.utils.rnn.pack_padded_sequence(batch, lengths, batch_first=True, enforce_sorted=enforce_sorted)


def pad_sequence(batch_seq: Union[tuple[Tensor, ...], list[Tensor]], enforce_sorted: bool = True) \
        -> tuple[Tensor, Tensor]:
    lengths = [seq.size(0) for seq in batch_seq]
    if not all_equal(lengths):
        # Sorting due to ``enforce_sorted = True`` default flag, for potential ONNX export.
        if enforce_sorted and not is_sorted(lengths, reverse=True):
            # https://stackoverflow.com/a/45514542/4907774
            lengths, batch_seq = map(list, sort_together([lengths, batch_seq], reverse=True))
        padded = torch.nn.utils.rnn.pad_sequence(batch_seq, batch_first=True)
        return padded, torch.as_tensor(lengths, device=padded.device)
    else:
        # Batch-first
        stacked = torch.stack(batch_seq)
        return stacked, torch.as_tensor(lengths, device=stacked.device)
