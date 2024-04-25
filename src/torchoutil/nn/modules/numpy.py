#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import torch
from torch import Tensor, nn

from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np

    from torchoutil.nn.functional.numpy import from_numpy, to_numpy

    class ToNumpy(nn.Module):
        """
        For more information, see :func:`~torchoutil.nn.functional.numpy.to_numpy`.
        """

        def __init__(self, *, dtype: Union[str, np.dtype, None] = None) -> None:
            super().__init__()
            self.dtype = dtype

        def forward(self, x: Union[Tensor, np.ndarray, list]) -> np.ndarray:
            return to_numpy(x, dtype=self.dtype)

    class FromNumpy(nn.Module):
        """
        For more information, see :func:`~torchoutil.nn.functional.numpy.from_numpy`.
        """

        def __init__(
            self,
            *,
            device: Union[str, torch.device, None] = None,
            dtype: Union[torch.dtype, None] = None,
        ) -> None:
            super().__init__()
            self.device = device
            self.dtype = dtype

        def forward(self, x: np.ndarray) -> Tensor:
            return from_numpy(x, dtype=self.dtype, device=self.device)
