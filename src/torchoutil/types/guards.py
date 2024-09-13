#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, List, Tuple

import torch
from torch import BoolTensor, Tensor
from typing_extensions import TypeGuard

from torchoutil.nn.functional.numpy import is_numpy_number_like, is_numpy_scalar_like
from torchoutil.pyoutil.typing import is_builtin_number, is_builtin_scalar
from torchoutil.types._typing import (
    BoolTensor1D,
    NumberLike,
    ScalarLike,
    Tensor0D,
    Tensor1D,
)


def is_tensor0d(x: Any) -> TypeGuard[Tensor0D]:
    """Returns True if x is a zero-dimensional torch Tensor."""
    return isinstance(x, Tensor) and x.ndim == 0


def is_number_like(x: Any) -> TypeGuard[NumberLike]:
    """Returns True if input is a scalar number.

    Accepted numbers-like objects are:
    - Python numbers (int, float, bool, complex)
    - Numpy zero-dimensional arrays
    - Numpy numbers
    - PyTorch zero-dimensional tensors
    """
    return is_builtin_number(x) or is_numpy_number_like(x) or is_tensor0d(x)


def is_scalar_like(x: Any) -> TypeGuard[ScalarLike]:
    """Returns True if input is a scalar number.

    Accepted scalar-like objects are:
    - Python scalars like (int, float, bool, complex, str, bytes, None)
    - Numpy zero-dimensional arrays
    - Numpy generic
    - PyTorch zero-dimensional tensors
    """
    return is_builtin_scalar(x) or is_numpy_scalar_like(x) or is_tensor0d(x)


def is_iterable_tensor(x: Any) -> TypeGuard[Iterable[Tensor]]:
    return isinstance(x, Iterable) and all(isinstance(xi, Tensor) for xi in x)


def is_list_tensor(x: Any) -> TypeGuard[List[Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x)


def is_tuple_tensor(x: Any) -> TypeGuard[Tuple[Tensor, ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, Tensor) for xi in x)


def is_integer_dtype(dtype: torch.dtype) -> bool:
    return not dtype.is_floating_point and not dtype.is_complex and dtype != torch.bool


def is_integer_tensor(x: Any) -> TypeGuard[Tensor]:
    return isinstance(x, Tensor) and is_integer_dtype(x.dtype)


def is_integer_tensor1d(x: Any) -> TypeGuard[Tensor1D]:
    return is_integer_tensor(x) and x.ndim == 1


def is_complex_tensor(x: Any) -> TypeGuard[Tensor]:
    return isinstance(x, Tensor) and x.is_complex()


def is_floating_tensor(x: Any) -> TypeGuard[Tensor]:
    return isinstance(x, Tensor) and x.is_floating_point()


def is_bool_tensor(x: Any) -> TypeGuard[BoolTensor]:
    return isinstance(x, BoolTensor)


def is_bool_tensor1d(x: Any) -> TypeGuard[BoolTensor1D]:
    return is_bool_tensor(x) and x.ndim == 1
