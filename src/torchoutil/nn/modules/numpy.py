#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from torch import Tensor

from torchoutil.core.make import DeviceLike, DTypeLike
from torchoutil.extras.numpy.definitions import np
from torchoutil.extras.numpy.functional import (
    numpy_to_tensor,
    tensor_to_numpy,
    to_numpy,
)
from torchoutil.nn.modules.module import Module


class ToNumpy(Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.numpy.to_numpy`.
    """

    def __init__(
        self,
        *,
        dtype: Union[str, np.dtype, None] = None,
        force: bool = False,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.force = force

    def forward(self, x: Union[Tensor, np.ndarray, list]) -> np.ndarray:
        return to_numpy(x, dtype=self.dtype, force=self.force)


class TensorToNumpy(Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.numpy.tensor_to_numpy`.
    """

    def __init__(
        self,
        *,
        dtype: Union[str, np.dtype, None] = None,
        force: bool = False,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.force = force

    def forward(self, x: Tensor) -> np.ndarray:
        return tensor_to_numpy(x, dtype=self.dtype, force=self.force)


class NumpyToTensor(Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.numpy.numpy_to_tensor`.
    """

    def __init__(
        self,
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: np.ndarray) -> Tensor:
        return numpy_to_tensor(x, dtype=self.dtype, device=self.device)
