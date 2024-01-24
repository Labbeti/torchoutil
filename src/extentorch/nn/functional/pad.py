#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, List, Sized, Union

import torch

from torch import Generator, Size, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from extentorch.nn.functional.get import get_device


PAD_ALIGNS = ("left", "right", "center", "random")


def pad_dim(
    x: Tensor,
    target_length: int,
    align: str = "left",
    pad_value: float = 0.0,
    dim: int = -1,
    mode: str = "constant",
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Generic function for pad a single dimension."""
    return pad_dims(x, [target_length], [align], pad_value, [dim], mode, generator)


def pad_dims(
    x: Tensor,
    target_lengths: Iterable[int],
    aligns: Iterable[str] = ("left",),
    pad_value: float = 0.0,
    dims: Iterable[int] = (-1,),
    mode: str = "constant",
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Generic function to pad multiple dimensions."""
    if isinstance(generator, int):
        generator = Generator().manual_seed(generator)

    target_lengths = list(target_lengths)
    aligns = list(aligns)
    dims = list(dims)

    if len(dims) == 0:
        raise ValueError(
            f"Invalid argument {dims=}. (cannot use an empty list of dimensions)"
        )

    if len(target_lengths) != len(dims):
        raise ValueError(
            f"Invalid number of targets lengths ({len(target_lengths)}) with the number of dimensions ({len(dims)})."
        )

    if len(aligns) != len(dims):
        raise ValueError(
            f"Invalid number of aligns ({len(aligns)}) with the number of dimensions ({len(dims)})."
        )

    pad_seq = __generate_pad_seq(x.shape, target_lengths, dims, aligns, generator)
    x = F.pad(x, pad_seq, mode=mode, value=pad_value)
    return x


def pad_and_stack_rec(
    sequence: Union[Tensor, int, float, tuple, list],
    pad_value: float,
    dtype: Union[None, torch.dtype] = None,
    device: Union[str, torch.device, None] = None,
) -> Tensor:
    """Recursive version of torch.nn.utils.rnn.pad_sequence, with padding of Tensors.

    Args:
        sequence: The sequence to pad. Must be convertable to tensor by having the correct number of dims in all sublists.
        pad_value: The pad value used.
        dtype: The dtype of the output Tensor. defaults to None.
        device: The device of the output Tensor. defaults to None.

    Example 1::
    ----------
        >>> sequence = [[1, 2], [3], [], [4, 5]]
        >>> output = pad_sequence_rec(sequence, 0)
        tensor([[1, 2], [3, 0], [0, 0], [4, 5]])

    Example 2::
    ----------
        >>> invalid_sequence = [[1, 2, 3], 3]
        >>> output = pad_sequence_rec(invalid_sequence, 0)
        ValueError : Cannot pad sequence of tensors of differents number of dims.

    """
    device = get_device(device)

    if isinstance(sequence, Tensor):
        return sequence.to(dtype=dtype, device=device)

    elif isinstance(sequence, (int, float)) or (
        isinstance(sequence, Sized) and len(sequence) == 0
    ):
        return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore

    elif isinstance(sequence, (list, tuple)):
        if all(isinstance(elt, (int, float)) for elt in sequence):
            return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore

        sequence = [
            pad_and_stack_rec(elt, pad_value, dtype, device) for elt in sequence
        ]
        # sequence is now a list[Tensor]
        shapes = [elt.shape for elt in sequence]

        # If all tensors have the same shape, just stack them
        if all(shape == shapes[0] for shape in shapes):
            return torch.stack(sequence)

        # If all tensors have the same number of dims
        elif all(elt.ndim == sequence[0].ndim for elt in sequence):
            if all(shape[1:] == shapes[0][1:] for shape in shapes):
                return pad_sequence(sequence, True, pad_value)
            else:
                max_lens = [
                    max(shape[i] for shape in shapes) for i in range(sequence[0].ndim)
                ]
                paddings = [
                    [
                        (max_lens[i] - elt.shape[i]) * j
                        for i in range(-1, -sequence[0].ndim, -1)
                        for j in range(2)
                    ]
                    for elt in sequence
                ]
                sequence = [
                    F.pad(elt, padding, value=pad_value)
                    for elt, padding in zip(sequence, paddings)
                ]
                return pad_sequence(sequence, True, pad_value)

        else:
            raise ValueError(
                f"Cannot pad sequence of tensors of differents number of dims. (with {sequence=} and {shapes=})"
            )

    else:
        raise TypeError(
            f"Invalid type {type(sequence)}. (expected Tensor, int, float, list or tuple)"
        )


def __generate_pad_seq(
    x_shape: Size,
    target_lengths: List[int],
    dims: List[int],
    aligns: List[str],
    generator: Union[None, Generator],
) -> List[int]:
    pad_seq = [0 for _ in range(len(x_shape) * 2)]
    for target_length, dim, align in zip(target_lengths, dims, aligns):
        missing = max(target_length - x_shape[dim], 0)

        if align == "left":
            missing_left = 0
            missing_right = missing
        elif align == "right":
            missing_left = missing
            missing_right = 0
        elif align == "center":
            missing_left = missing // 2 + missing % 2
            missing_right = missing // 2
        elif align == "random":
            missing_left = int(
                torch.randint(
                    low=0, high=missing + 1, size=(), generator=generator
                ).item()
            )
            missing_right = missing - missing_left
        else:
            raise ValueError(
                f"Invalid argument {align=}. (expected one of {PAD_ALIGNS})"
            )

        # Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
        idx = len(x_shape) - (dim % len(x_shape)) - 1
        assert pad_seq[idx * 2] == 0 and pad_seq[idx * 2 + 1] == 0
        pad_seq[idx * 2] = missing_left
        pad_seq[idx * 2 + 1] = missing_right

    return pad_seq
