#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, List, Literal, Union

import torch
from torch import Generator, Tensor

from torchoutil.nn.functional.get import get_generator

CROP_ALIGNS = ("left", "right", "center", "random")
CropAlign = Literal["left", "right", "center", "random"]


def crop_dim(
    x: Tensor,
    target_length: int,
    *,
    align: CropAlign = "left",
    dim: int = -1,
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Generic function to crop a single dimension."""
    return crop_dims(
        x,
        [target_length],
        aligns=[align],
        dims=[dim],
        generator=generator,
    )


def crop_dims(
    x: Tensor,
    target_lengths: Iterable[int],
    *,
    aligns: Union[CropAlign, Iterable[CropAlign]] = "left",
    dims: Union[Iterable[int], Literal["auto"]] = "auto",
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
        raise ValueError(
            f"Invalid number of targets lengths ({len(target_lengths)}) with the number of dimensions ({len(dims)})."
        )

    if len(aligns_lst) != len(dims):
        raise ValueError(
            f"Invalid number of aligns ({len(aligns_lst)}) with the number of dimensions ({len(dims)})."
        )

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
            raise ValueError(
                f"Invalid argument {align=}. (expected one of {CROP_ALIGNS})"
            )

        slices[dim] = slice(start, end)

    x = x[slices]
    return x
