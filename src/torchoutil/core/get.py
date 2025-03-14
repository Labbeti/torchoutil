#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch
from torch import Generator

from torchoutil.pyoutil.logging import warn_once

# For backward compatibility only
from .make import (  # noqa: F401
    CUDA_IF_AVAILABLE,
    DeviceLike,
    DTypeLike,
    GeneratorLike,
    make_device,
    make_dtype,
    make_generator,
)


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
