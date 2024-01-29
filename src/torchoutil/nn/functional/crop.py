#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

import torch
from torch import Generator, Tensor

CROP_ALIGNS = ("left", "right", "center", "random")


def crop_dim(
    x: Tensor,
    target_length: int,
    align: str = "left",
    dim: int = -1,
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Generic function to crop a single dimension."""
    return crop_dims(x, [target_length], [align], [dim], generator)


def crop_dims(
    x: Tensor,
    target_lengths: Iterable[int],
    aligns: Iterable[str] = ("left",),
    dims: Iterable[int] = (-1,),
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Generic function to crop multiple dimensions."""

    target_lengths = list(target_lengths)
    aligns = list(aligns)
    dims = list(dims)

    if isinstance(generator, int):
        generator = Generator().manual_seed(generator)

    if len(target_lengths) != len(dims):
        raise ValueError(
            f"Invalid number of targets lengths ({len(target_lengths)}) with the number of dimensions ({len(dims)})."
        )

    if len(aligns) != len(dims):
        raise ValueError(
            f"Invalid number of aligns ({len(aligns)}) with the number of dimensions ({len(dims)})."
        )

    slices = [slice(None)] * len(x.shape)

    for target_length, dim, align in zip(target_lengths, dims, aligns):
        if x.shape[dim] <= target_length:
            continue

        if align == "left":
            start = 0
            end = target_length
        elif align == "right":
            start = x.shape[dim] - target_length
            end = None
        elif align == "center":
            diff = x.shape[dim] - target_length
            start = diff // 2 + diff % 2
            end = start + target_length
        elif align == "random":
            diff = x.shape[dim] - target_length
            start = torch.randint(low=0, high=diff, size=(), generator=generator).item()
            end = start + target_length
        else:
            raise ValueError(
                f"Invalid argument {align=}. (expected one of {CROP_ALIGNS})"
            )

        slices[dim] = slice(start, end)

    x = x[slices]
    return x
