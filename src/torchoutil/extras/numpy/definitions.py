#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from torchoutil.core.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy  # noqa: F401  # type: ignore
    import numpy as np  # type: ignore

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

else:
    from torchoutil.extras.numpy import _numpy_placeholder as np  # noqa: F401
    from torchoutil.extras.numpy import _numpy_placeholder as numpy

    ACCEPTED_NUMPY_DTYPES = ()


NumpyNumberLike = Union[np.ndarray, np.number]
NumpyScalarLike = Union[np.ndarray, np.generic]
