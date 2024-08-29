#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch
from torch import Generator

from torchoutil.types.classes import (
    CUDA_IF_AVAILABLE,
    DeviceLike,
    DTypeLike,
    GeneratorLike,
)
from torchoutil.types.dtype_typing import (
    DTypeEnum,
    enum_dtype_to_torch_dtype,
    str_to_torch_dtype,
)

_DEVICE_CUDA_IF_AVAILABLE = CUDA_IF_AVAILABLE  # for backward compatibility only


def get_device(
    device: DeviceLike = CUDA_IF_AVAILABLE,
) -> Optional[torch.device]:
    if device == CUDA_IF_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, (str, int)):
        device = torch.device(device)
    return device


def get_dtype(
    dtype: DTypeLike = None,
) -> Optional[torch.dtype]:
    if dtype == "default":
        dtype = torch.get_default_dtype()
    elif isinstance(dtype, DTypeEnum):
        dtype = enum_dtype_to_torch_dtype(dtype)
    elif isinstance(dtype, str):
        dtype = str_to_torch_dtype(dtype)
    return dtype


def get_generator(
    generator: GeneratorLike = None,
) -> Optional[Generator]:
    if isinstance(generator, int):
        generator = Generator().manual_seed(generator)
    elif generator == "default":
        generator = torch.default_generator
    return generator
