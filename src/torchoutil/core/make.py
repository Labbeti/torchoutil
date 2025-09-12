#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Optional, Union, overload

import torch
from typing_extensions import TypeAlias

from .dtype_enum import DTypeEnum, enum_dtype_to_torch_dtype, str_to_torch_dtype

DeviceLike: TypeAlias = Union[
    torch.device, None, Literal["default", "cuda_if_available"], str, int
]
DTypeLike: TypeAlias = Union[torch.dtype, None, Literal["default"], str, DTypeEnum]
GeneratorLike: TypeAlias = Union[torch.Generator, None, Literal["default"], int]

CUDA_IF_AVAILABLE = "cuda_if_available"


def get_default_device() -> torch.device:
    """Returns default device used when creating a tensor."""
    return torch.empty((0,)).device


def get_default_dtype() -> torch.dtype:
    return torch.get_default_dtype()


def get_default_generator() -> torch.Generator:
    return torch.default_generator


def set_default_dtype(dtype: DTypeLike) -> None:
    dtype = as_dtype(dtype)
    torch.set_default_dtype(dtype)


def set_default_generator(generator: GeneratorLike) -> None:
    generator = as_generator(generator)
    if generator is not None:
        torch.default_generator.set_state(generator.get_state())


@overload
def as_device(device: Literal[None]) -> None:
    ...


@overload
def as_device(
    device: Union[str, int, torch.device] = CUDA_IF_AVAILABLE,
) -> torch.device:
    ...


def as_device(device: DeviceLike = CUDA_IF_AVAILABLE) -> Optional[torch.device]:
    """Create concrete device object from device-like object."""
    if isinstance(device, (torch.device, type(None))) or device is ...:
        return device
    elif device == "default":
        return get_default_device()
    elif device == CUDA_IF_AVAILABLE:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, (str, int)):
        return torch.device(device)
    else:
        msg = f"Invalid argument type {type(device)}. (expected torch.device, None, str, int or {CUDA_IF_AVAILABLE})"
        raise TypeError(msg)


@overload
def as_dtype(dtype: Literal[None] = None) -> None:
    ...


@overload
def as_dtype(dtype: Union[str, DTypeEnum, torch.dtype]) -> torch.dtype:
    ...


def as_dtype(dtype: DTypeLike = None) -> Optional[torch.dtype]:
    """Create concrete dtype object from dtype-like object."""
    if isinstance(dtype, (torch.dtype, type(None))) or dtype is ...:
        return dtype
    elif dtype == "default":
        return get_default_dtype()
    elif isinstance(dtype, DTypeEnum):
        return enum_dtype_to_torch_dtype(dtype)
    elif isinstance(dtype, str):
        return str_to_torch_dtype(dtype)
    else:
        msg = f"Invalid argument type {type(dtype)}. (expected torch.dtype, None, str or torchoutil.DTypeEnum)"
        raise TypeError(msg)


@overload
def as_generator(generator: Literal[None] = None) -> None:
    ...


@overload
def as_generator(
    generator: Union[int, torch.Generator, Literal["default"]],
) -> torch.Generator:
    ...


def as_generator(generator: GeneratorLike = None) -> Optional[torch.Generator]:
    """Create concrete generator object from generator-like object."""
    if isinstance(generator, (torch.Generator, type(None))) or generator is ...:
        return generator
    elif isinstance(generator, int):
        return torch.Generator().manual_seed(generator)
    elif generator == "default":
        return get_default_generator()
    else:
        msg = f"Invalid argument type {type(generator)}. (expected torch.Generator, None, int or 'default')"
        raise TypeError(msg)
