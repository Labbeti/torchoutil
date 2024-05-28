#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Optional, Union

import torch
from torch import Generator
from torch.types import Device

_DEVICE_CUDA_IF_AVAILABLE = "cuda_if_available"


def get_device(
    device: Union[Device, Literal["cuda_if_available"]] = _DEVICE_CUDA_IF_AVAILABLE,
) -> Optional[torch.device]:
    if device == _DEVICE_CUDA_IF_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)
    return device


def get_generator(
    generator: Union[int, Generator, None] = None,
) -> Optional[Generator]:
    if isinstance(generator, int):
        generator = Generator().manual_seed(generator)
    return generator
