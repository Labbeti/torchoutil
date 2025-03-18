#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Optional, Union

import torch
from torch import Generator
from torch.types import Device
from typing_extensions import TypeAlias

from .dtype_enum import DTypeEnum, enum_dtype_to_torch_dtype, str_to_torch_dtype

DeviceLike: TypeAlias = Union[Device, Literal["default", "cuda_if_available"]]
DTypeLike: TypeAlias = Union[torch.dtype, None, Literal["default"], str, DTypeEnum]
GeneratorLike: TypeAlias = Union[int, Generator, None, Literal["default"]]

CUDA_IF_AVAILABLE = "cuda_if_available"


def get_default_device() -> torch.device:
    """Returns default device used when creating a tensor."""
    return torch.empty((0,)).device


def get_default_dtype() -> torch.dtype:
    return torch.get_default_dtype()


def get_default_generator() -> Generator:
    return torch.default_generator


def make_device(device: DeviceLike = CUDA_IF_AVAILABLE) -> Optional[torch.device]:
    """Create concrete device object from device-like object."""
    if isinstance(device, (torch.device, type(None))):
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


def make_dtype(dtype: DTypeLike = None) -> Optional[torch.dtype]:
    """Create concrete dtype object from dtype-like object."""
    if isinstance(dtype, (torch.dtype, type(None))):
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


def make_generator(generator: GeneratorLike = None) -> Optional[Generator]:
    """Create concrete generator object from generator-like object."""
    if isinstance(generator, (Generator, type(None))):
        return generator
    elif isinstance(generator, int):
        return Generator().manual_seed(generator)
    elif generator == "default":
        return get_default_generator()
    else:
        msg = f"Invalid argument type {type(generator)}. (expected torch.Generator, None, int or 'default')"
        raise TypeError(msg)
