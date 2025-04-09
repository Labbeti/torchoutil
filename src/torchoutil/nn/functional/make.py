#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.core.make import (  # noqa: F401
    CUDA_IF_AVAILABLE,
    DeviceLike,
    DTypeLike,
    GeneratorLike,
    as_device,
    as_dtype,
    as_generator,
    get_default_device,
    get_default_dtype,
    get_default_generator,
    set_default_dtype,
    set_default_generator,
)
