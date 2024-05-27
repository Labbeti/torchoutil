#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, Iterable, List, Literal, Sized, Tuple, Union

import torch
from torch import Generator, Size, Tensor
from torch.nn import functional as F
from torch.types import Device, Number

from torchoutil.nn.functional.get import get_device
from torchoutil.nn.functional.others import can_be_stacked
from torchoutil.utils.type_checks import is_scalar

PAD_ALIGNS = ("left", "right", "center", "random")
PadAlign = Literal["left", "right", "center", "random"]
PadValue = Union[Number, Callable[[Tensor], Number]]


def pad_dim(
    x: Tensor,
    target_length: int,
    align: PadAlign = "left",
    pad_value: PadValue = 0.0,
    dim: int = -1,
    mode: str = "constant",
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Generic function for pad a single dimension."""
    return pad_dims(x, [target_length], [align], pad_value, [dim], mode, generator)


def pad_dims(
    x: Tensor,
    target_lengths: Iterable[int],
    aligns: Iterable[PadAlign] = ("left",),
    pad_value: PadValue = 0.0,
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

    if isinstance(pad_value, Callable):
        pad_value = pad_value(x)

    pad_seq = __generate_pad_seq(x.shape, target_lengths, dims, aligns, generator)
    x = F.pad(x, pad_seq, mode=mode, value=pad_value)
    return x


def pad_and_stack_rec(
    sequence: Union[Tensor, int, float, tuple, list],
    pad_value: Number,
    *,
    device: Device = None,
    dtype: Union[None, torch.dtype] = None,
) -> Tensor:
    """Recursive version of torch.nn.utils.rnn.pad_sequence, with padding of Tensors.

    Args:
        sequence: The sequence to pad. Must be convertable to tensor by having the correct number of dims in all sublists.
        pad_value: The pad value used.
        dtype: The dtype of the output Tensor. defaults to None.
        device: The device of the output Tensor. defaults to None.

    Example 1::
    -----------
        >>> sequence = [[1, 2], [3], [], [4, 5]]
        >>> output = pad_sequence_rec(sequence, 0)
        tensor([[1, 2], [3, 0], [0, 0], [4, 5]])

    Example 2::
    -----------
        >>> invalid_sequence = [[1, 2, 3], 3]
        >>> output = pad_sequence_rec(invalid_sequence, 0)
        ValueError : Cannot pad sequence of tensors of differents number of dims.

    """
    device = get_device(device)

    if isinstance(sequence, Tensor):
        return sequence.to(dtype=dtype, device=device)

    elif is_scalar(sequence) or (isinstance(sequence, Sized) and len(sequence) == 0):
        return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore

    elif isinstance(sequence, (list, tuple)):
        sequence = [
            pad_and_stack_rec(elt, pad_value, dtype=dtype, device=device)
            for elt in sequence
        ]
        if can_be_stacked(sequence):
            return torch.stack(sequence)

        shapes = [elt.shape for elt in sequence]
        shape0 = shapes[0]

        if not all(len(shape) == len(shape0) for shape in shapes):
            raise ValueError(
                f"Cannot pad sequence of tensors of differents number of dims. (with {shapes=})"
            )

        max_lens = [max(shape[i] for shape in shapes) for i in range(len(shape0))]
        sequence = [
            pad_dims(
                xi,
                target_lengths=max_lens,
                pad_value=pad_value,
                aligns=["left"] * xi.ndim,
                dims=range(xi.ndim),
            )
            for xi in sequence
        ]
        result = torch.stack(sequence)
        return result

    else:
        raise TypeError(
            f"Invalid type {type(sequence)}. (expected Tensor, int, float, list or tuple)"
        )


def cat_padded_batch(
    x1: Tensor,
    x1_lens: Tensor,
    x2: Tensor,
    x2_lens: Tensor,
    seq_dim: int = -1,
    batch_dim: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Concatenate padded batched of sequences.

    Args:
        x1: First batch with D dims of shape (batch_size, ..., N1, ...)
        x1_lens: First lengths of each element in sequence dim of shape (batch_size,).
        x2: Second batch with D dims of shape (batch_size, ..., N2, ...)
            The shape must be the same than x1 unless for the dimension N2.
        x2_lens: Second lengths of each element in sequence dim of shape (batch_size,).
        seq_dim: Dimension index of sequence. defaults to -1.
        batch_dim: Batch dimension index. defaults to 0.
    """
    _check_cat_padded_batch(x1, x1_lens, x2, x2_lens, seq_dim, batch_dim)
    x12_lens = x1_lens + x2_lens
    sum_size_12 = x1.shape[seq_dim] + x2.shape[seq_dim]

    x12 = pad_dim(x1, sum_size_12, dim=seq_dim)
    kwd: Dict[str, Any] = dict(device=x1.device, dtype=torch.long)
    indices = torch.arange(x2_lens.max().item(), **kwd)

    unsq_x1_lens = x1_lens
    ndim = x1.ndim
    for i in range(ndim):
        if i != (seq_dim % ndim):
            indices = indices.unsqueeze(dim=i)
        if i != (batch_dim % ndim):
            unsq_x1_lens = unsq_x1_lens.unsqueeze(dim=i)

    expand_size = list(x2.shape)
    expand_size[seq_dim] = -1
    indices = indices.expand(*expand_size)
    indices = indices + unsq_x1_lens
    x12.scatter_(seq_dim, indices, x2)

    max_size_12 = int(x12_lens.max().item())
    if max_size_12 < sum_size_12:
        slices = [slice(None) for _ in range(ndim)]
        slices[seq_dim] = slice(max_size_12)
        x12 = x12[slices]

    return x12, x12_lens


def __generate_pad_seq(
    x_shape: Size,
    target_lengths: List[int],
    dims: List[int],
    aligns: List[PadAlign],
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


def _check_cat_padded_batch(
    x1: Tensor,
    x1_lens: Tensor,
    x2: Tensor,
    x2_lens: Tensor,
    seq_dim: int,
    batch_dim: int,
) -> None:
    if x1.ndim != x2.ndim:
        raise ValueError(f"Invalid arguments ndims. (found {x1.ndim=} != {x2.ndim=})")
    if x1.ndim < 2:
        raise ValueError(f"Invalid arguments ndims. (found {x1.ndim=} < 2)")

    batch_size = x1.shape[batch_dim]
    if not (x1_lens.shape == x2_lens.shape == Size((batch_size,))):
        raise ValueError(
            f"Invalid arguments shape. (with {x1_lens.shape=} and {x2_lens.shape=})"
        )

    x1_shape = torch.as_tensor(x1.shape)
    x2_shape = torch.as_tensor(x2.shape)
    eq_mask = x1_shape.eq(x2_shape)
    eq_mask[seq_dim] = True
    if not eq_mask.all():
        raise ValueError(f"Invalid arguments shape. (with {x1.shape=} and {x2.shape=})")
