#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

import torch
from torch import Tensor

from torchoutil.nn.functional.pad import pad_dim
from torchoutil.types import BoolTensor, LongTensor


def extract_segments(x: Tensor) -> LongTensor:
    """
    Example 1
    ----------
    >>> x = torch.as_tensor([0, 1, 1, 0, 0, 1, 1, 1, 1, 0]).bool()
    >>> starts, ends = extract_segments(x)
    >>> starts
    ... tensor([1, 5])
    >>> ends
    ... tensor([3, 9])

    Example 2
    ----------
    >>> x = torch.as_tensor([[1, 1, 1, 0], [1, 0, 0, 1]]).bool()
    >>> indices, starts, ends = extract_segments(x)
    >>> indices
    ... tensor([0, 1, 1])
    >>> starts
    ... tensor([0, 0, 3])
    >>> ends
    ... tensor([3, 1, 4])

    Args:
        x: (..., N) bool tensor containing D dims

    Returns:
        segments: (D+1, M) tensor, where M is the total number of segments
            When D > 1, segments also contains indices of the source column for each start and end value. See Example 2 for detail.
    """
    if not isinstance(x, BoolTensor):
        raise ValueError(f"Invalid argument {x=}. (expected BoolTensor)")

    x = x.int()
    x = pad_dim(x, x.shape[-1] + 2, align="center", pad_value=0, dim=-1)
    diff = x[..., 1:] - x[..., :-1]

    starts = torch.stack(torch.where(diff > 0))
    ends = torch.stack(torch.where(diff < 0))

    result = torch.cat([starts, ends[-1:]], dim=0)
    return result  # type: ignore


def segments_to_list(
    segments: Tensor,
    maxsize: Union[int, Tuple[int, ...], None],
) -> list:
    if segments.shape[0] == 2:
        starts, ends = segments.tolist()
        return list(zip(starts, ends))

    elif segments.shape[0] in (0, 1):
        msg = f"Invalid argument shape {segments.shape=}. (expected first dim >= 2)"
        raise ValueError(msg)

    if maxsize is None:
        num_elems = segments[0].max().item() + 1
        next_maxsize = None
    elif isinstance(maxsize, tuple):
        num_elems = maxsize[0]
        next_maxsize = maxsize[1:]
    else:
        num_elems = maxsize
        next_maxsize = None

    indices = torch.arange(num_elems)
    result = [
        segments_to_list(segments[1:, ..., segments[0] == idx], maxsize=next_maxsize)
        for idx in indices
    ]
    return result
