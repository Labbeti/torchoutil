#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, List, Tuple, Union

import torch
from torch import Tensor

from torchoutil.core.make import DeviceLike, as_device
from torchoutil.nn import functional as F
from torchoutil.nn.functional.padding import pad_and_stack_rec, pad_dim
from torchoutil.pyoutil.warnings import deprecated_alias
from torchoutil.types import BoolTensor, LongTensor


def activity_to_segments(x: Tensor) -> LongTensor:
    """Extracts segments start and end positions from a boolean activity/mask tensor.

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
            When D > 1, segments also contains indices of the source column for each start and end value. See Example 2 for details.
    """
    if not isinstance(x, BoolTensor):
        msg = f"Invalid argument {x=}. (expected BoolTensor)"
        raise ValueError(msg)

    x = x.int()
    x = pad_dim(x, x.shape[-1] + 2, align="center", pad_value=0, dim=-1)
    diff = x[..., 1:] - x[..., :-1]

    starts = torch.stack(torch.where(diff > 0))
    ends = torch.stack(torch.where(diff < 0))

    result = torch.cat([starts, ends[-1:]], dim=0)
    return result  # type: ignore


def segments_to_segments_list(
    segments: Tensor,
    maxsize: Union[int, Tuple[int, ...], None] = None,
) -> Union[List[Tuple[int, int]], list]:
    """Converts segments starts and ends to a list of (start, end) positions."""
    if segments.shape[0] in (0, 1):
        msg = f"Invalid argument shape {segments.shape=}. (expected first dim >= 2)"
        raise ValueError(msg)

    elif segments.shape[0] == 2:
        starts, ends = segments.tolist()
        return list(zip(starts, ends))

    if maxsize is None:
        num_elems = segments[0].max().item() + 1
        next_maxsize = None
    elif isinstance(maxsize, tuple):
        num_elems = maxsize[0]
        next_maxsize = maxsize[1:]
    else:
        num_elems = maxsize
        next_maxsize = None

    arange = torch.arange(num_elems)
    result = [
        segments_to_segments_list(
            segments[1:, ..., segments[0] == idx], maxsize=next_maxsize
        )
        for idx in arange
    ]
    return result


def segments_list_to_activity(
    segments_list: Union[List[Tuple[int, int]], Tensor, list],
    maxsize: Union[int, None] = None,
    device: DeviceLike = None,
) -> BoolTensor:
    if device is None and isinstance(segments_list, Tensor):
        device = segments_list.device
    else:
        device = as_device(device)

    if F.ndim(segments_list) == 2 or (
        F.ndim(segments_list) == 1 and len(segments_list) == 0
    ):
        if len(segments_list) == 0:
            if maxsize is None:
                num_elems = 0
            else:
                num_elems = maxsize

            return F.full((num_elems,), False, dtype=torch.bool, device=device)  # type: ignore

        starts, ends = F.as_tensor(segments_list).transpose(0, 1)

        if maxsize is None:
            num_elems = ends.max().item()
        else:
            num_elems = maxsize

        arange = F.arange(num_elems, device=device)[None]
        activity = (starts[:, None] <= arange) & (arange < ends[:, None])
        activity = activity.any(dim=0)
        return activity  # type: ignore

    elif isinstance(segments_list, Iterable):
        activities = [
            segments_list_to_activity(segments_list_i)  # type: ignore
            for segments_list_i in segments_list
        ]
        return pad_and_stack_rec(activities, False)  # type: ignore

    else:
        msg = f"Invalid argument type {type(segments_list)}."
        raise TypeError(msg)


def activity_to_segments_list(x: Tensor) -> Union[List[Tuple[int, int]], list]:
    segments = activity_to_segments(x)
    segments_lst = segments_to_segments_list(segments, x.shape[-1])
    return segments_lst


def segments_to_activity(x: Tensor) -> BoolTensor:
    segments_lst = segments_to_segments_list(x, x.shape[-1])
    activity = segments_list_to_activity(segments_lst, x.shape[-1])
    return activity


@deprecated_alias(activity_to_segments)
def extract_segments(*args, **kwargs):
    ...


@deprecated_alias(segments_to_segments_list)
def segments_to_list(*args, **kwargs):
    ...
