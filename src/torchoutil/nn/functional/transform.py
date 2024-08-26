#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, Tensor

from pyoutil.collections import flatten as builtin_flatten
from pyoutil.functools import identity  # noqa: F401
from pyoutil.typing import TBuiltinScalar
from torchoutil.nn.functional.crop import crop_dim
from torchoutil.nn.functional.get import get_generator
from torchoutil.nn.functional.pad import PadMode, PadValue, pad_dim
from torchoutil.types import is_scalar_like, np
from torchoutil.types._hints import Tensor1D

T = TypeVar("T")
U = TypeVar("U")

PadCropAlign = Literal["left", "right", "center", "random"]
PAD_CROP_ALIGN_VALUES = ("left", "right", "center", "random")


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


def resample_nearest_rates(
    x: Tensor,
    rates: Union[float, Iterable[float]],
    *,
    dims: Union[int, Iterable[int]] = -1,
    round_fn: Callable[[Tensor], Tensor] = torch.floor,
) -> Tensor:
    """Nearest neigbour resampling using tensor slices.

    Args:
        x: Input tensor.
        rate: The reduction factor of each axis, e.g. a factor of 0.5 will divide the input axes by 2.
        dims: Dimensions to apply resampling. defaults to -1.
        round_fn: Rounding function to compute sub-indices. defaults to torch.floor.
    """
    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    if isinstance(rates, (int, float)):
        steps = [1.0 / rates] * len(dims)
    else:
        steps = [1.0 / rate for rate in rates]

    return resample_nearest_steps(
        x,
        steps,
        dims=dims,
        round_fn=round_fn,
    )


def resample_nearest_freqs(
    x: Tensor,
    orig_freq: int,
    new_freq: int,
    *,
    dims: Union[int, Iterable[int]] = -1,
    round_fn: Callable[[Tensor], Tensor] = torch.floor,
) -> Tensor:
    return resample_nearest_steps(
        x,
        orig_freq / new_freq,
        dims=dims,
        round_fn=round_fn,
    )


def resample_nearest_steps(
    x: Tensor,
    steps: Union[float, Iterable[float]],
    *,
    dims: Union[int, Iterable[int]] = -1,
    round_fn: Callable[[Tensor], Tensor] = torch.floor,
) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    if isinstance(steps, (int, float)):
        steps = [steps] * len(dims)
    else:
        steps = list(steps)  # type: ignore
        if len(steps) != len(dims):
            raise ValueError(f"Invalid arguments sizes {len(steps)=} != {len(dims)}.")

    slices: List[Union[slice, Tensor]] = [slice(None)] * x.ndim

    for dim, step in zip(dims, steps):
        length = x.shape[dim]
        indexes = torch.arange(0, length, step)
        indexes = round_fn(indexes).long().clamp(min=0, max=length - 1)
        slices[dim] = indexes

    x = x[slices]
    return x


def transform_drop(
    transform: Callable[[T], T],
    x: T,
    p: float,
    generator: Union[int, Generator, None] = None,
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
    generator = get_generator(generator)

    p_floor = math.floor(p)
    for _ in range(p_floor):
        x = transform(x)

    sampled = torch.rand((), generator=generator)
    if sampled + p_floor < p:
        x = transform(x)

    return x


def pad_and_crop_dim(
    x: Tensor,
    target_length: int,
    *,
    align: PadCropAlign = "left",
    pad_value: PadValue = 0.0,
    dim: int = -1,
    mode: PadMode = "constant",
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Pad and crop along the specified dimension."""
    x = pad_dim(
        x,
        target_length,
        align=align,
        pad_value=pad_value,
        dim=dim,
        mode=mode,
        generator=generator,
    )
    x = crop_dim(
        x,
        target_length,
        align=align,
        dim=dim,
        generator=generator,
    )
    return x


def shuffled(
    x: Tensor,
    dims: Union[int, Iterable[int]] = -1,
    generator: Union[int, Generator, None] = None,
) -> Tensor:
    """Returns a shuffled version of the input tensor along specific dimension(s)."""
    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    generator = get_generator(generator)
    slices: List[Union[slice, Tensor]] = [slice(None) for _ in range(x.ndim)]
    for dim in dims:
        indices = torch.randperm(x.shape[dim], generator=generator)
        slices[dim] = indices
    x = x[slices]
    return x


@overload
def flatten(
    x: Tensor,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> Tensor1D:
    ...


@overload
def flatten(
    x: Union[np.ndarray, np.generic],
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> np.ndarray:
    ...


@overload
def flatten(
    x: TBuiltinScalar,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[TBuiltinScalar]:
    ...


@overload
def flatten(
    x: Iterable[TBuiltinScalar],
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[TBuiltinScalar]:
    ...


@overload
def flatten(
    x: Any,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[Any]:
    ...


def flatten(
    x: Any,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> Any:
    if isinstance(x, Tensor):
        end_dim = end_dim if end_dim is not None else x.ndim - 1
        return x.flatten(start_dim, end_dim)
    elif (
        isinstance(x, (np.ndarray, np.generic))
        and start_dim == 0
        and (end_dim is None or end_dim >= x.ndim - 1)
    ):
        return x.flatten()
    else:
        return builtin_flatten(x, start_dim, end_dim, is_scalar_fn=is_scalar_like)
