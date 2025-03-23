#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import Tensor
from typing_extensions import TypeGuard, TypeIs

from torchoutil.extras.numpy import is_numpy_number_like, is_numpy_scalar_like, np
from torchoutil.pyoutil.functools import function_alias
from torchoutil.pyoutil.typing import (
    is_builtin_number,
    is_builtin_scalar,
    isinstance_guard,
)
from torchoutil.pyoutil.warnings import deprecated_function

from ._typing import (
    BoolTensor,
    BoolTensor1D,
    ComplexFloatingTensor,
    FloatingTensor,
    IntegralTensor,
    IntegralTensor1D,
    NumberLike,
    ScalarLike,
    Tensor0D,
    TensorOrArray,
)


def is_number_like(x: Any) -> TypeGuard[NumberLike]:
    """Returns True if input is a scalar number.

    Accepted numbers-like objects are:
    - Python numbers (int, float, bool, complex)
    - Numpy zero-dimensional arrays
    - Numpy numbers
    - PyTorch zero-dimensional tensors
    """
    return is_builtin_number(x) or is_numpy_number_like(x) or isinstance(x, Tensor0D)


def is_scalar_like(x: Any) -> TypeGuard[ScalarLike]:
    """Returns True if input is a scalar number.

    Accepted scalar-like objects are:
    - Python scalars like (int, float, bool, complex, None, str, bytes)
    - Numpy zero-dimensional arrays
    - Numpy generic
    - PyTorch zero-dimensional tensors
    """
    return is_builtin_scalar(x) or is_numpy_scalar_like(x) or isinstance(x, Tensor0D)


def is_tensor_or_array(x: Any) -> TypeIs[TensorOrArray]:
    return isinstance(x, (Tensor, np.ndarray))


@function_alias(is_tensor_or_array)
def is_tensor_like(*args, **kwargs):
    ...


def is_integral_dtype(dtype: torch.dtype) -> bool:
    return is_integral_tensor(torch.empty((0,), dtype=dtype))


@deprecated_function("{fn_name}, use `isinstance(x, to.BoolTensor)` instead.")
def is_bool_tensor(x: Any) -> TypeIs[BoolTensor]:
    return isinstance(x, BoolTensor)


@deprecated_function("{fn_name}, use `isinstance(x, to.BoolTensor1D)` instead.")
def is_bool_tensor1d(x: Any) -> TypeIs[BoolTensor1D]:
    return isinstance(x, BoolTensor1D)


@deprecated_function(
    "{fn_name}, use `isinstance(x, to.ComplexFloatingTensor)` instead."
)
def is_complex_tensor(x: Any) -> TypeIs[ComplexFloatingTensor]:
    return isinstance(x, ComplexFloatingTensor)


@deprecated_function(
    "{fn_name}, use `to.isinstance_guard(x, Dict[str, Tensor])` instead."
)
def is_dict_str_tensor(x: Any) -> TypeIs[Dict[str, Tensor]]:
    return isinstance_guard(x, Dict[str, Tensor])


@deprecated_function("{fn_name}, use `isinstance(x, to.FloatingTensor)` instead.")
def is_floating_tensor(x: Any) -> TypeIs[FloatingTensor]:
    return isinstance(x, FloatingTensor)


@deprecated_function("{fn_name}, use `isinstance(x, to.IntegralTensor)` instead.")
def is_integral_tensor(x: Any) -> TypeIs[IntegralTensor]:
    return isinstance(x, IntegralTensor)


@deprecated_function("{fn_name}, use `isinstance(x, to.IntegralTensor1D)` instead.")
def is_integral_tensor1d(x: Any) -> TypeIs[IntegralTensor1D]:
    return isinstance(x, IntegralTensor1D)


@deprecated_function(
    "{fn_name}, use `to.isinstance_guard(x, Iterable[Tensor])` instead."
)
def is_iterable_tensor(x: Any) -> TypeIs[Iterable[Tensor]]:
    return isinstance_guard(x, Iterable[Tensor])


@deprecated_function("{fn_name}, use `to.isinstance_guard(x, List[Tensor])` instead.")
def is_list_tensor(x: Any) -> TypeIs[List[Tensor]]:
    return isinstance_guard(x, List[Tensor])


@deprecated_function("{fn_name}, use `isinstance(x, Tensor0D)` instead.")
def is_tensor0d(x: Any) -> TypeIs[Tensor0D]:
    return isinstance(x, Tensor0D)


@deprecated_function(
    "{fn_name}, use `to.isinstance_guard(x, Tuple[Tensor, ...])` instead."
)
def is_tuple_tensor(x: Any) -> TypeIs[Tuple[Tensor, ...]]:
    return isinstance_guard(x, Tuple[Tensor, ...])
