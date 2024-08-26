#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Optional, Union

import torch
from torch import Generator
from torch.types import Device

CUDA_IF_AVAILABLE = "cuda_if_available"
_DEVICE_CUDA_IF_AVAILABLE = CUDA_IF_AVAILABLE  # for backward compatibility only


def get_device(
    device: Union[Device, Literal["cuda_if_available"]] = CUDA_IF_AVAILABLE,
) -> Optional[torch.device]:
    if device == CUDA_IF_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, (str, int)):
        device = torch.device(device)
    return device


def get_dtype(
    dtype: Union[torch.dtype, None, Literal["default"]] = None,
) -> Optional[torch.dtype]:
    if dtype == "default":
        dtype = torch.get_default_dtype()
    return dtype


def get_generator(
    generator: Union[int, Generator, None, Literal["default"]] = None,
) -> Optional[Generator]:
    if isinstance(generator, int):
        generator = Generator().manual_seed(generator)
    elif generator == "default":
        generator = torch.default_generator
    return generator
