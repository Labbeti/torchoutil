#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, List, Tuple

import torch
from torch import Tensor
from typing_extensions import TypeIs

from torchoutil.extras.numpy import is_numpy_number_like, is_numpy_scalar_like, np
from torchoutil.pyoutil.typing import is_builtin_number, is_builtin_scalar
from torchoutil.types._typing import (
    BoolTensor,
    BoolTensor1D,
    ComplexFloatingTensor,
    FloatingTensor,
    NumberLike,
    ScalarLike,
    SignedIntegerTensor,
    SignedIntegerTensor1D,
    Tensor0D,
    TensorLike,
)


def is_bool_tensor(x: Any) -> TypeIs[BoolTensor]:
    return isinstance(x, BoolTensor)


def is_bool_tensor1d(x: Any) -> TypeIs[BoolTensor1D]:
    return is_bool_tensor(x) and x.ndim == 1


def is_complex_tensor(x: Any) -> TypeIs[ComplexFloatingTensor]:
    return isinstance(x, Tensor) and x.is_complex()


def is_floating_tensor(x: Any) -> TypeIs[FloatingTensor]:
    return isinstance(x, Tensor) and x.is_floating_point()


def is_integral_dtype(dtype: torch.dtype) -> bool:
    return not dtype.is_floating_point and not dtype.is_complex and dtype.is_signed


def is_integral_tensor(x: Any) -> TypeIs[SignedIntegerTensor]:
    return isinstance(x, Tensor) and is_integral_dtype(x.dtype)


def is_integral_tensor1d(x: Any) -> TypeIs[SignedIntegerTensor1D]:
    return is_integral_tensor(x) and x.ndim == 1


def is_iterable_tensor(x: Any) -> TypeIs[Iterable[Tensor]]:
    return isinstance(x, Iterable) and all(isinstance(xi, Tensor) for xi in x)


def is_list_tensor(x: Any) -> TypeIs[List[Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x)


def is_number_like(x: Any) -> TypeIs[NumberLike]:
    """Returns True if input is a scalar number.

    Accepted numbers-like objects are:
    - Python numbers (int, float, bool, complex)
    - Numpy zero-dimensional arrays
    - Numpy numbers
    - PyTorch zero-dimensional tensors
    """
    return is_builtin_number(x) or is_numpy_number_like(x) or is_tensor0d(x)


def is_scalar_like(x: Any) -> TypeIs[ScalarLike]:
    """Returns True if input is a scalar number.

    Accepted scalar-like objects are:
    - Python scalars like (int, float, bool, complex, None, str, bytes)
    - Numpy zero-dimensional arrays
    - Numpy generic
    - PyTorch zero-dimensional tensors
    """
    return is_builtin_scalar(x) or is_numpy_scalar_like(x) or is_tensor0d(x)


def is_tensor0d(x: Any) -> TypeIs[Tensor0D]:
    """Returns True if x is a zero-dimensional torch Tensor."""
    return isinstance(x, Tensor) and x.ndim == 0


def is_tensor_like(x: Any) -> TensorLike:
    return isinstance(x, (Tensor, np.ndarray))


def is_tuple_tensor(x: Any) -> TypeIs[Tuple[Tensor, ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, Tensor) for xi in x)
