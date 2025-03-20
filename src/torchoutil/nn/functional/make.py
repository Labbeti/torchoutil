#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.core.make import (  # noqa: F401
    CUDA_IF_AVAILABLE,
    DeviceLike,
    DTypeLike,
    GeneratorLike,
    get_default_device,
    get_default_dtype,
    get_default_generator,
    make_device,
    make_dtype,
    make_generator,
)
