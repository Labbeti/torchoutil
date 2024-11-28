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
    Sequence,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, Tensor

from torchoutil.core.get import (
    DeviceLike,
    DTypeLike,
    get_device,
    get_dtype,
    get_generator,
)
from torchoutil.extras.numpy import np
from torchoutil.nn.functional.crop import crop_dim
from torchoutil.nn.functional.pad import PadMode, PadValue, pad_dim
from torchoutil.pyoutil.collections import all_eq
from torchoutil.pyoutil.collections import flatten as builtin_flatten
from torchoutil.pyoutil.collections import prod as builtin_prod
from torchoutil.pyoutil.functools import identity  # noqa: F401
from torchoutil.pyoutil.typing import T_BuiltinScalar
from torchoutil.types import is_number_like, is_scalar_like
from torchoutil.types._typing import (
    BuiltinNumber,
    NumberLike,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
)

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
        msg = f"Function repeat_interleave_nd does not supports 0-d tensors. (found {x.ndim=} == 0)"
        raise ValueError(msg)

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
    x: T_BuiltinScalar,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[T_BuiltinScalar]:
    ...


@overload
def flatten(
    x: Iterable[T_BuiltinScalar],
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[T_BuiltinScalar]:
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
    elif isinstance(x, np.generic):
        return x.flatten()
    elif isinstance(x, np.ndarray):
        if start_dim == 0 and (end_dim is None or end_dim >= x.ndim - 1):
            return x.flatten()
        else:
            end_dim = end_dim if end_dim is not None else x.ndim - 1
            shape = list(x.shape)
            shape = (
                shape[:start_dim]
                + [builtin_prod(shape[start_dim : end_dim + 1])]
                + shape[end_dim + 1 :]
            )
            return x.reshape(*shape)
    else:
        return builtin_flatten(x, start_dim, end_dim, is_scalar_fn=is_scalar_like)


@overload
def to_tensor(
    data: BuiltinNumber,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor0D:
    ...


@overload
def to_tensor(
    data: Sequence[BuiltinNumber],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor1D:
    ...


@overload
def to_tensor(
    data: Sequence[Sequence[BuiltinNumber]],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor2D:
    ...


@overload
def to_tensor(
    data: Sequence[Sequence[Sequence[BuiltinNumber]]],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor3D:
    ...


@overload
def to_tensor(data: Any, dtype: DTypeLike = None, device: DeviceLike = None) -> Tensor:
    ...


def to_tensor(data: Any, dtype: DTypeLike = None, device: DeviceLike = None) -> Tensor:
    """Convert arbitrary data to tensor. Unlike `torch.as_tensor`, it works recursively and stack sequences like List[Tensor].

    Args:
        data: Data to convert to tensor. Can be Tensor, np.ndarray, list, tuple or any number-like object.
        dtype: Target torch dtype.
        device: Target torch device.

    Returns:
        PyTorch tensor created from data.
    """
    if isinstance(data, (Tensor, np.ndarray)) or is_number_like(data):
        dtype = get_dtype(dtype)
        device = get_device(device)
        return torch.as_tensor(data, dtype=dtype, device=device)

    elif isinstance(data, (list, tuple)):
        tensors = [to_tensor(data_i, dtype=dtype, device=device) for data_i in data]
        shapes = [tensor.shape for tensor in tensors]
        if not all_eq(shapes):
            msg = f"Cannot convert to tensor a list of elements with heterogeneous shapes. (found {shapes})"
            raise ValueError(msg)
        return torch.stack(tensors)

    else:
        EXPECTED = (Tensor, np.ndarray, NumberLike, list, tuple)
        msg = f"Invalid argument type '{type(data)}'. (expected one of {EXPECTED})"
        raise TypeError(msg)
