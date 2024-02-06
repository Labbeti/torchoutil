#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from torch import Tensor

from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np

    def to_numpy(
        x: Union[Tensor, np.ndarray, list],
        dtype: Union[str, np.dtype, None] = None,
    ) -> np.ndarray:
        if isinstance(x, Tensor):
            return x.cpu().numpy().astype(dtype=dtype)
        else:
            return np.asarray(x, dtype=dtype)
