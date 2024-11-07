#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, List, Literal, Union

import torch
from torch import Generator, Tensor

from torchoutil.core.get import get_generator

CROP_ALIGNS = ("left", "right", "center", "random")
CropAlign = Literal["left", "right", "center", "random"]


def crop_dim(
    x: Tensor,
    target_length: int,
    *,
    dim: int = -1,
    align: CropAlign = "left",
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Generic function to crop a single dimension."""
    return crop_dims(
        x,
        [target_length],
        dims=[dim],
        aligns=[align],
        generator=generator,
    )


def crop_dims(
    x: Tensor,
    target_lengths: Iterable[int],
    *,
    dims: Union[Iterable[int], Literal["auto"]] = "auto",
    aligns: Union[CropAlign, Iterable[CropAlign]] = "left",
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Generic function to crop multiple dimensions."""

    target_lengths = list(target_lengths)

    aligns_lst: List[CropAlign]
    if isinstance(aligns, str):
        aligns_lst = [aligns] * len(target_lengths)
    else:
        aligns_lst = list(aligns)
    del aligns

    if dims == "auto":
        dims = list(range(-len(target_lengths), 0))
    else:
        dims = list(dims)

    generator = get_generator(generator)

    if len(target_lengths) != len(dims):
        msg = f"Invalid number of targets lengths ({len(target_lengths)}) with the number of dimensions ({len(dims)})."
        raise ValueError(msg)

    if len(aligns_lst) != len(dims):
        msg = f"Invalid number of aligns ({len(aligns_lst)}) with the number of dimensions ({len(dims)})."
        raise ValueError(msg)

    slices = [slice(None)] * len(x.shape)

    for target_length, dim, align in zip(target_lengths, dims, aligns_lst):
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
            msg = f"Invalid argument {align=}. (expected one of {CROP_ALIGNS})"
            raise ValueError(msg)

        slices[dim] = slice(start, end)

    x = x[slices]
    return x
