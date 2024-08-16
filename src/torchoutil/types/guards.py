#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, List, Tuple

import torch
from torch import BoolTensor, Tensor
from typing_extensions import TypeGuard

from pyoutil.typing import is_builtin_number
from torchoutil.types._hints import (
    BoolTensor1D,
    NumberLike,
    NumpyNumberLike,
    Tensor0D,
    Tensor1D,
)
from torchoutil.types.classes import np


def is_numpy_number_like(x: Any) -> TypeGuard[NumpyNumberLike]:
    """Returns True if x is an instance of a numpy number type or a zero-dimensional numpy array.
    If numpy is not installed, this function always returns False.
    """
    return isinstance(x, (np.number, np.bool_)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def is_tensor0d(x: Any) -> TypeGuard[Tensor0D]:
    """Returns True if x is a zero-dimensional torch Tensor."""
    return isinstance(x, Tensor) and x.ndim == 0


def is_scalar(x: Any) -> TypeGuard[NumberLike]:
    """Returns True if input is a scalar number.

    Accepted scalars are:
    - Python numbers (int, float, bool, complex)
    - PyTorch zero-dimensional tensors
    - Numpy zero-dimensional arrays
    - Numpy number scalars
    """
    return is_builtin_number(x) or is_numpy_number_like(x) or is_tensor0d(x)


def is_iterable_tensor(x: Any) -> TypeGuard[Iterable[Tensor]]:
    return isinstance(x, Iterable) and all(isinstance(xi, Tensor) for xi in x)


def is_list_tensor(x: Any) -> TypeGuard[List[Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x)


def is_tuple_tensor(x: Any) -> TypeGuard[Tuple[Tensor, ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, Tensor) for xi in x)


def is_integer_dtype(dtype: torch.dtype) -> bool:
    return not dtype.is_floating_point and not dtype.is_complex and dtype != torch.bool


def is_integer_tensor(x: Tensor) -> TypeGuard[Tensor]:
    return isinstance(x, Tensor) and is_integer_dtype(x.dtype)


def is_integer_tensor1d(x: Tensor) -> TypeGuard[Tensor1D]:
    return is_integer_tensor(x) and x.ndim == 1


def is_complex_tensor(x: Tensor) -> TypeGuard[Tensor]:
    return isinstance(x, Tensor) and x.is_complex()


def is_floating_tensor(x: Tensor) -> TypeGuard[Tensor]:
    return isinstance(x, Tensor) and x.is_floating_point()


def is_bool_tensor(x: Tensor) -> TypeGuard[BoolTensor]:
    return isinstance(x, BoolTensor)


def is_bool_tensor1d(x: Tensor) -> TypeGuard[BoolTensor1D]:
    return is_bool_tensor(x) and x.ndim == 1
