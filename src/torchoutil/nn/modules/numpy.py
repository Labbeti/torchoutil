#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Union

import torch
from torch import Tensor, nn
from torch.types import Device

from torchoutil.nn.functional.numpy import numpy_to_tensor, tensor_to_numpy, to_numpy
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if not _NUMPY_AVAILABLE:

    class np:
        dtype = Any
        ndarray = Any

else:
    import numpy as np


class ToNumpy(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.numpy.to_numpy`.
    """

    def __init__(self, *, dtype: Union[str, np.dtype, None] = None) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, x: Union[Tensor, np.ndarray, list]) -> np.ndarray:
        return to_numpy(x, dtype=self.dtype)


class TensorToNumpy(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.numpy.tensor_to_numpy`.
    """

    def __init__(self, *, dtype: Union[str, np.dtype, None] = None) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, x: Union[Tensor, np.ndarray, list]) -> np.ndarray:
        return tensor_to_numpy(x, dtype=self.dtype)


class NumpyToTensor(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.numpy.numpy_to_tensor`.
    """

    def __init__(
        self,
        *,
        device: Device = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: np.ndarray) -> Tensor:
        return numpy_to_tensor(x, dtype=self.dtype, device=self.device)
