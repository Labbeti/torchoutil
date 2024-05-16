#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import torch
from torch import Tensor
from torch.types import Device

from torchoutil.nn.functional.get import get_device
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if not _NUMPY_AVAILABLE:
    ACCEPTED_NUMPY_DTYPES = ()

else:
    import numpy as np

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

    def to_numpy(
        x: Union[Tensor, np.ndarray, list],
        *,
        dtype: Union[str, np.dtype, None] = None,
    ) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(x, Tensor):
            return tensor_to_numpy(x, dtype=dtype)
        else:
            return np.asarray(x, dtype=dtype)

    def tensor_to_numpy(
        x: Tensor,
        *,
        dtype: Union[str, np.dtype, None] = None,
    ) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return x.cpu().numpy().astype(dtype=dtype)

    def numpy_to_tensor(
        x: np.ndarray,
        *,
        device: Device = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> Tensor:
        """Convert numpy array to PyTorch tensor."""
        device = get_device(device)
        return torch.from_numpy(x).to(dtype=dtype, device=device)
