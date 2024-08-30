#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence, Union

import torch
from torch import Tensor
from torch.types import Device

from torchoutil.nn.functional.get import get_device
from torchoutil.pyoutil import get_current_fn_name
from torchoutil.types import np
from torchoutil.utils.packaging import torch_version_ge_1_13


def to_numpy(
    x: Union[Tensor, np.ndarray, Sequence],
    *,
    dtype: Union[str, np.dtype, None] = None,
    force: bool = False,
) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, Tensor):
        return tensor_to_numpy(x, dtype=dtype, force=force)
    else:
        return np.asarray(x, dtype=dtype)


def tensor_to_numpy(
    x: Tensor,
    *,
    dtype: Union[str, np.dtype, None] = None,
    force: bool = False,
) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    if torch_version_ge_1_13():
        kwargs = dict(force=force)
    elif not force:
        kwargs = dict()
    else:
        raise ValueError(
            f"Invalid argument {force=} for {get_current_fn_name()}. (expected True because torrch version is below 1.13)"
        )

    x_arr: np.ndarray = x.cpu().numpy(**kwargs)
    if dtype is not None:
        x_arr = x_arr.astype(dtype=dtype)
    return x_arr


def numpy_to_tensor(
    x: np.ndarray,
    *,
    device: Device = None,
    dtype: Union[torch.dtype, None] = None,
) -> Tensor:
    """Convert numpy array to PyTorch tensor."""
    device = get_device(device)
    return torch.from_numpy(x).to(dtype=dtype, device=device)
