#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal, Union

import torch
from torch import Generator
from torch.types import Device

from torchoutil.types.dtype_typing import DTypeEnum
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

DeviceLike = Union[Device, Literal["cuda_if_available"]]
DTypeLike = Union[torch.dtype, None, Literal["default"], str, DTypeEnum]
GeneratorLike = Union[int, Generator, None, Literal["default"]]

CUDA_IF_AVAILABLE = "cuda_if_available"


if not _NUMPY_AVAILABLE:
    from torchoutil.types import _numpy_placeholder as np  # noqa: F401
    from torchoutil.types import _numpy_placeholder as numpy

    ACCEPTED_NUMPY_DTYPES = ()

else:
    import numpy  # noqa: F401
    import numpy as np

    # Numpy dtypes that can be converted to tensor
    ACCEPTED_NUMPY_DTYPES = (
        np.float64,
        np.float32,
        np.float16,
        np.complex64,
        np.complex128,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint8,
        bool,
    )
