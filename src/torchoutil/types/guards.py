#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import Tensor
from typing_extensions import TypeGuard, TypeIs

from torchoutil.extras.numpy import is_numpy_number_like, is_numpy_scalar_like, np
from torchoutil.pyoutil.typing import (
    is_builtin_number,
    is_builtin_scalar,
    isinstance_guard,
)
from torchoutil.types._typing import (
    BoolTensor,
    BoolTensor1D,
    ComplexFloatingTensor,
    FloatingTensor,
    NumberLike,
    ScalarLike,
    Tensor0D,
    Tensor1D,
    TensorOrArray,
)


def is_bool_tensor(x: Any) -> TypeIs[BoolTensor]:
    """Deprecated: Use `isinstance(x, to.BoolTensor)` instead."""
    return isinstance(x, BoolTensor)


def is_bool_tensor1d(x: Any) -> TypeIs[BoolTensor1D]:
    """Deprecated: Use `isinstance(x, to.BoolTensor1D)` instead."""
    return isinstance(x, BoolTensor1D)


def is_complex_tensor(x: Any) -> TypeIs[ComplexFloatingTensor]:
    """Deprecated: Use `isinstance(x, to.ComplexFloatingTensor)` instead."""
    return isinstance(x, ComplexFloatingTensor)


def is_dict_str_tensor(x: Any) -> TypeIs[Dict[str, Tensor]]:
    """Deprecated: Use `to.isinstance_guard(x, Dict[str, Tensor])` instead."""
    return isinstance_guard(x, Dict[str, Tensor])


def is_floating_tensor(x: Any) -> TypeIs[FloatingTensor]:
    """Deprecated: Use `isinstance(x, to.FloatingTensor)` instead."""
    return isinstance(x, FloatingTensor)


def is_integral_dtype(dtype: torch.dtype) -> bool:
    return not dtype.is_floating_point and not dtype.is_complex


def is_integral_tensor(x: Any) -> TypeIs[Tensor]:
    return isinstance(x, Tensor) and is_integral_dtype(x.dtype)


def is_integral_tensor1d(x: Any) -> TypeIs[Tensor1D]:
    return is_integral_tensor(x) and x.ndim == 1


def is_iterable_tensor(x: Any) -> TypeIs[Iterable[Tensor]]:
    """Deprecated: Use `to.isinstance_guard(x, Iterable[Tensor])` instead."""
    return isinstance_guard(x, Iterable[Tensor])


def is_list_tensor(x: Any) -> TypeIs[List[Tensor]]:
    """Deprecated: Use `to.isinstance_guard(x, List[Tensor])` instead."""
    return isinstance_guard(x, List[Tensor])


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
    - Python scalars like (int, float, bool, complex, None, str, bytes)
    - Numpy zero-dimensional arrays
    - Numpy generic
    - PyTorch zero-dimensional tensors
    """
    return is_builtin_scalar(x) or is_numpy_scalar_like(x) or is_tensor0d(x)


def is_tensor0d(x: Any) -> TypeIs[Tensor0D]:
    """Deprecated: Use `isinstance(x, to.Tensor0D)` instead.

    Returns True if x is a zero-dimensional torch Tensor."""
    return isinstance(x, Tensor0D)


def is_tensor_like(x: Any) -> TypeIs[TensorOrArray]:
    return is_tensor_or_array(x)


def is_tensor_or_array(x: Any) -> TypeIs[TensorOrArray]:
    return isinstance(x, (Tensor, np.ndarray))


def is_tuple_tensor(x: Any) -> TypeIs[Tuple[Tensor, ...]]:
    """Deprecated: Use `to.isinstance_guard(x, Tuple[Tensor, ...])` instead."""
    return isinstance_guard(x, Tuple[Tensor, ...])
