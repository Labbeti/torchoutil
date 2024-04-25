#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
from typing import Callable, Iterable, List, TypeVar, Union

import torch
from torch import Tensor

T = TypeVar("T")


def repeat_interleave_nd(x: Tensor, repeats: int, dim: int = 0) -> Tensor:
    """Generalized version of torch.repeat_interleave for N >= 1 dimensions.
    The output size will be (..., D*repeats, ...), where D is the size of the dimension of the dim argument.

    Args:
        x: Any tensor of shape (..., D, ...) with at least 1 dim.
        repeats: Number of repeats.
        dim: The dimension to repeat. defaults to 0.

    Examples::
    ----------
        >>> x = torch.as_tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        >>> repeat_interleave_nd(x, n=2, dim=0)
        tensor([[0, 1, 2, 3],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [4, 5, 6, 7]])
    """
    if x.ndim == 0:
        raise ValueError(
            f"Function repeat_interleave_nd does not supports 0-d tensors. (found {x.ndim=} == 0)"
        )

    dim = dim % x.ndim
    x = x.unsqueeze(dim=dim + 1)
    shape = list(x.shape)
    shape[dim + 1] = repeats
    x = x.expand(*shape)
    x = x.flatten(dim, dim + 1)
    return x


def resample_nearest(
    x: Tensor,
    rates: Union[float, Iterable[float]],
    dims: Union[int, Iterable[int]] = -1,
    round_fn: Callable[[Tensor], Tensor] = torch.floor,
) -> Tensor:
    """Nearest resampling using a rate.

    Args:
        x: Input tensor.
        rate: The
    """
    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    if isinstance(rates, (float, int)):
        rates = [rates] * len(dims)
    else:
        rates = list(rates)  # type: ignore
        if len(rates) != len(dims):
            raise ValueError(f"Invalid arguments sizes {len(rates)=} != {len(dims)}.")

    slices: List[Union[slice, Tensor]] = [slice(None)] * len(x.shape)
    for dim, rate in zip(dims, rates):
        length = x.shape[dim]
        step = 1.0 / rate
        indexes = torch.arange(0, length, step)
        indexes = round_fn(indexes).long().clamp(min=0, max=length - 1)
        slices[dim] = indexes

    x = x[slices]
    return x


def transform_drop(
    transform: Callable[[T], T],
    x: T,
    p: float,
) -> T:
    """Apply a transform on a tensor with a probability of p.

    Args:
        transform: Transform to apply.
        x: Argument of the transform.
        p: Probability p to apply the transform. Cannot be negative.
            If > 1, it will apply the transform floor(p) times and apply a last time with a probability of p - floor(p).
    """
    if p < 0.0:
        raise ValueError(f"Invalid argument {p=} < 0")

    p_floor = math.floor(p)
    for _ in range(p_floor):
        x = transform(x)

    if random.random() + p_floor < p:
        x = transform(x)

    return x
