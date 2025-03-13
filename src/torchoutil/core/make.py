#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Optional, Union

import torch
from torch import Generator
from torch.types import Device

from torchoutil.pyoutil.logging import warn_once

from .dtype_enum import DTypeEnum, enum_dtype_to_torch_dtype, str_to_torch_dtype

DeviceLike = Union[Device, Literal["default", "cuda_if_available"]]
DTypeLike = Union[torch.dtype, None, Literal["default"], str, DTypeEnum]
GeneratorLike = Union[int, Generator, None, Literal["default"]]

CUDA_IF_AVAILABLE = "cuda_if_available"


def get_default_device() -> torch.device:
    """Returns default device used when creating a tensor."""
    return torch.empty((0,)).device


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
        return torch.get_default_dtype()
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
        return torch.default_generator
    else:
        msg = f"Invalid argument type {type(generator)}. (expected torch.Generator, None, int or 'default')"
        raise TypeError(msg)


def get_device(device: DeviceLike = CUDA_IF_AVAILABLE) -> Optional[torch.device]:
    """DEPRECATED: Use make_device instead.

    Create concrete device object from device-like object."""
    warn_once("Deprecated function get_device. Use make_device instead.")
    return make_device(device)


def get_dtype(dtype: DTypeLike = None) -> Optional[torch.dtype]:
    """DEPRECATED: Use make_dtype instead.

    Create concrete dtype object from dtype-like object."""
    warn_once("Deprecated function get_dtype. Use make_dtype instead.")
    return make_dtype(dtype)


def get_generator(generator: GeneratorLike = None) -> Optional[Generator]:
    """DEPRECATED: Use make_generator instead.

    Create concrete generator object from generator-like object."""
    warn_once("Deprecated function get_generator. Use make_generator instead.")
    return make_generator(generator)
