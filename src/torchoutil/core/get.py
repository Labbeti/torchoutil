#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Optional, Union

import torch
from torch import Generator
from torch.types import Device

from .dtype_enum import DTypeEnum, enum_dtype_to_torch_dtype, str_to_torch_dtype

DeviceLike = Union[Device, Literal["cuda_if_available"]]
DTypeLike = Union[torch.dtype, None, Literal["default"], str, DTypeEnum]
GeneratorLike = Union[int, Generator, None, Literal["default"]]

CUDA_IF_AVAILABLE = "cuda_if_available"


def get_device(device: DeviceLike = CUDA_IF_AVAILABLE) -> Optional[torch.device]:
    if isinstance(device, (torch.device, type(None))):
        return device
    elif device == CUDA_IF_AVAILABLE:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, (str, int)):
        device = torch.device(device)
    else:
        msg = f"Invalid argument type {type(device)}. (expected torch.device, None, str or int)"
        raise TypeError(msg)


def get_dtype(dtype: DTypeLike = None) -> Optional[torch.dtype]:
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


def get_generator(generator: GeneratorLike = None) -> Optional[Generator]:
    if isinstance(generator, (Generator, type(None))):
        return generator
    elif isinstance(generator, int):
        return Generator().manual_seed(generator)
    elif generator == "default":
        return torch.default_generator
    else:
        msg = f"Invalid argument type {type(generator)}. (expected torch.Generator, None, int or 'default')"
        raise TypeError(msg)
