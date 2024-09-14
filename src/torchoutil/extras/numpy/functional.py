#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Sequence, Union

import torch
from torch import Tensor
from typing_extensions import TypeGuard

from torchoutil.core.get import DeviceLike, DTypeLike, get_device, get_dtype
from torchoutil.core.packaging import _NUMPY_AVAILABLE, torch_version_ge_1_13
from torchoutil.pyoutil import get_current_fn_name


if _NUMPY_AVAILABLE:
    import numpy  # noqa: F401  # type: ignore
    import numpy as np  # type: ignore

    # Numpy dtypes that can be converted to tensor
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

else:
    from torchoutil.core import _numpy_placeholder as np  # noqa: F401
    from torchoutil.core import _numpy_placeholder as numpy

    ACCEPTED_NUMPY_DTYPES = ()


NumpyNumberLike = Union[np.ndarray, np.number]
NumpyScalarLike = Union[np.ndarray, np.generic]


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
    if x.dtype == np.complex64:
        float_dtype = np.float32
    elif x.dtype == np.complex128:
        float_dtype = np.float64
    elif x.dtype == np.complex256:
        float_dtype = np.float128
    else:
        DTYPES = (np.complex64, np.complex128, np.complex256)
        raise ValueError(f"Unexpected dtype {x.dtype}. (expected one of {DTYPES})")

    return x.view(float_dtype).reshape(*x.shape, 2)


def numpy_view_as_complex(x: np.ndarray) -> np.ndarray:
    return x[..., 0] + x[..., 1] * 1j


def numpy_is_floating_point(x: Union[np.ndarray, np.generic]) -> bool:
    return isinstance(x, np.floating)


def numpy_is_complex(x: Union[np.ndarray, np.generic]) -> bool:
    return np.iscomplexobj(x)  # type: ignore


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
