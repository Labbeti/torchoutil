#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Literal, Union

import torch
from torch import Tensor
from typing_extensions import NoReturn, TypeGuard

from torchoutil.nn.functional.get import get_device
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np

    def to_numpy(
        x: Union[Tensor, np.ndarray, list],
        *,
        dtype: Union[str, np.dtype, None] = None,
    ) -> np.ndarray:
        if isinstance(x, Tensor):
            return x.cpu().numpy().astype(dtype=dtype)
        else:
            return np.asarray(x, dtype=dtype)

    def numpy_to_tensor(
        x: np.ndarray,
        *,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> Tensor:
        device = get_device(device)
        return torch.from_numpy(x).to(dtype=dtype, device=device)

    def is_numpy_scalar(x: Any) -> TypeGuard[Union[np.generic, np.ndarray]]:
        """Returns True if x is a numpy generic type or a zero-dimensional numpy array."""
        return isinstance(x, np.generic) or (isinstance(x, np.ndarray) and x.ndim == 0)

else:
    _NUMPY_UNAVAIBLE_MESSAGE = (
        "Cannot call function '{fn}' because numpy package is not installed."
        "Please install it using 'pip install numpy' or 'pip install torchoutil[extras]'."
    )

    def to_numpy(
        x: Any,
        *,
        dtype: Any = None,
    ) -> NoReturn:
        raise RuntimeError(_NUMPY_UNAVAIBLE_MESSAGE.format(fn="to_numpy"))

    def numpy_to_tensor(
        x: Any,
        *,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> NoReturn:
        raise RuntimeError(_NUMPY_UNAVAIBLE_MESSAGE.format(fn="numpy_to_tensor"))

    def is_numpy_scalar(x: Any) -> Literal[False]:
        return False
