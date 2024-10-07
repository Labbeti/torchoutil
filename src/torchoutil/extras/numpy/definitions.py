#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from torchoutil.core.packaging import _NUMPY_AVAILABLE

if not _NUMPY_AVAILABLE:
    from torchoutil.extras.numpy import _numpy_fallback as np  # noqa: F401
    from torchoutil.extras.numpy import _numpy_fallback as numpy

    ACCEPTED_NUMPY_DTYPES = ()

else:
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


NumpyNumberLike = Union[np.ndarray, np.number]
NumpyScalarLike = Union[np.ndarray, np.generic]
