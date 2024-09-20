#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Sequence, Union

import torch
from torch import Tensor
from typing_extensions import TypeGuard

from torchoutil.core.get import DeviceLike, DTypeLike, get_device, get_dtype
from torchoutil.core.packaging import torch_version_ge_1_13
from torchoutil.extras.numpy.definitions import NumpyNumberLike, NumpyScalarLike, np
from torchoutil.pyoutil import get_current_fn_name


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
        return np.asarray(x, dtype=dtype)  # type: ignore


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
        msg = f"Invalid argument {force=} for {get_current_fn_name()}. (expected True because torch version is below 1.13)"
        raise ValueError(msg)

    x_arr: np.ndarray = x.cpu().numpy(**kwargs)
    if dtype is not None:  # supports older numpy version
        x_arr = x_arr.astype(dtype=dtype)  # type: ignore
    return x_arr


def numpy_to_tensor(
    x: Union[np.ndarray, np.number],
    *,
    device: DeviceLike = None,
    dtype: DTypeLike = None,
) -> Tensor:
    """Convert numpy array to PyTorch tensor."""
    device = get_device(device)
    dtype = get_dtype(dtype)
    return torch.from_numpy(x).to(dtype=dtype, device=device)


def numpy_view_as_real(x: np.ndarray) -> np.ndarray:
    """Convert complex array to float array.

    Args:
        x: The input complex array of any shape (...,)
    Returns:
        x_real: The same data in a float array of shape (..., 2)
    """
    assert numpy_is_complex(x)
    float_dtype = numpy_complex_dtype_to_float_dtype(x.dtype)
    if x.ndim > 0:
        return x.view(float_dtype).reshape(*x.shape, 2)
    else:
        # note: rebuild array here because view does not work on 0d arrays
        return np.array([x.real, x.imag], dtype=float_dtype)


def numpy_complex_dtype_to_float_dtype(dtype: np.dtype) -> np.dtype:
    """Returns the associated float dtype from complex dtype. If input dtype is not complex, it just returns the same dtype."""
    return np.empty((0,), dtype=dtype).real.dtype


def numpy_view_as_complex(x: np.ndarray) -> np.ndarray:
    """Convert complex array to float array.

    Args:
        x: The input float array of any shape (..., 2)
    Returns:
        x_real: The same data in a complex array of shape (...,)
    """
    assert not numpy_is_complex(x)
    return x[..., 0] + x[..., 1] * 1j


def numpy_is_floating_point(x: Union[np.ndarray, np.generic]) -> bool:
    return isinstance(x, np.floating)


def numpy_is_complex(x: Union[np.ndarray, np.generic]) -> bool:
    return np.iscomplexobj(x)


def numpy_is_complex_dtype(dtype: np.dtype) -> bool:
    return np.iscomplexobj(np.empty((0,), dtype=dtype))


def is_numpy_number_like(x: Any) -> TypeGuard[NumpyNumberLike]:
    """Returns True if x is an instance of a numpy number type, a np.bool_ or a zero-dimensional numpy array.
    If numpy is not installed, this function always returns False.
    """
    return isinstance(x, (np.number, np.bool_)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def is_numpy_scalar_like(x: Any) -> TypeGuard[NumpyScalarLike]:
    """Returns True if x is an instance of a numpy number type or a zero-dimensional numpy array.
    If numpy is not installed, this function always returns False.
    """
    return isinstance(x, np.generic) or (isinstance(x, np.ndarray) and x.ndim == 0)
