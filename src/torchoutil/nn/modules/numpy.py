#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import torch
from torch import Tensor, nn

from torchoutil.nn.functional.numpy import to_numpy
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np

    class ToNumpy(nn.Module):
        def __init__(self, dtype: Union[str, np.dtype, None] = None) -> None:
            super().__init__()
            self.dtype = dtype

        def forward(self, x: Union[Tensor, np.ndarray, list]) -> np.ndarray:
            return to_numpy(x, self.dtype)

    class FromNumpy(nn.Module):
        def forward(self, x: np.ndarray) -> Tensor:
            return torch.from_numpy(x)
