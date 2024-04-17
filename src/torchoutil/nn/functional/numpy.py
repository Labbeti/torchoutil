#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Union

import torch
from torch import Tensor
from typing_extensions import TypeGuard

from torchoutil.nn.functional.get import get_device
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

    def from_numpy(
        x: np.ndarray,
        dtype: Union[torch.dtype, None] = None,
        device: Union[str, torch.device, None] = None,
    ) -> Tensor:
        device = get_device(device)
        return torch.from_numpy(x).to(dtype=dtype, device=device)

    def is_numpy_scalar(x: Any) -> TypeGuard[np.generic]:
        return np.isscalar(x) and x.__class__.__module__ != "__builtin__"
