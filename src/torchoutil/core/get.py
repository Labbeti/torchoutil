#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.pyoutil.warnings import deprecated_alias

# For backward compatibility only
from .make import CUDA_IF_AVAILABLE  # noqa: F401
from .make import DeviceLike  # noqa: F401
from .make import DTypeLike  # noqa: F401
from .make import GeneratorLike  # noqa: F401
from .make import as_device, as_dtype, as_generator  # noqa: F401


@deprecated_alias(as_device)
def get_device(*args, **kwargs):
    ...


@deprecated_alias(as_dtype)
def get_dtype(*args, **kwargs):
    ...


@deprecated_alias(as_generator)
def get_generator(*args, **kwargs):
    ...
