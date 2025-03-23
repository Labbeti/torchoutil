#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, List, Sized, Tuple, TypeVar, Union, overload

import torch
from typing_extensions import TypeGuard

from torchoutil.extras.numpy import (
    ACCEPTED_NUMPY_DTYPES,
    np,
    numpy_all_eq,
    numpy_all_ne,
    numpy_is_complex,
    numpy_is_floating_point,
)
from torchoutil.nn.functional.others import nelement
from torchoutil.pyoutil.collections import all_eq as builtin_all_eq
from torchoutil.pyoutil.collections import all_ne as builtin_all_ne
from torchoutil.pyoutil.collections import is_sorted as builtin_is_sorted
from torchoutil.pyoutil.functools import function_alias
from torchoutil.pyoutil.typing import is_builtin_number, isinstance_guard
from torchoutil.pyoutil.warnings import deprecated_alias
from torchoutil.types._typing import (
    ComplexFloatingTensor,
    FloatingTensor,
    ScalarLike,
    T_TensorOrArray,
    Tensor0D,
    TensorOrArray,
)
from torchoutil.types.guards import is_scalar_like
from torchoutil.types.tensor_subclasses import Tensor

T = TypeVar("T")
U = TypeVar("U")


def is_stackable(
    tensors: Union[List[Any], Tuple[Any, ...]],
) -> TypeGuard[Union[List[Tensor], Tuple[Tensor, ...]]]:
    """Returns True if inputs can be passed to `torch.stack` function, i.e. contains a list or tuple of tensors with the same shape."""
    if not isinstance_guard(tensors, List[Tensor]) and not isinstance_guard(
        tensors, Tuple[Tensor, ...]
    ):
        return False
    if len(tensors) == 0:
        return False
    shape0 = tensors[0].shape
    result = all(tensor.shape == shape0 for tensor in tensors[1:])
    return result


def is_convertible_to_tensor(x: Any) -> bool:
    """Returns True if inputs can be passed to `torch.as_tensor` function.

    This function returns False for heterogeneous inputs like `[[], 1]`, but this kind of value can be accepted by `torch.as_tensor`.
    """
    if isinstance(x, Tensor):
        return True
    else:
        return __can_be_converted_to_tensor_nested(x)


def __can_be_converted_to_tensor_list_tuple(x: Union[List, Tuple]) -> bool:
    if len(x) == 0:
        return True

    valid_items = all(__can_be_converted_to_tensor_nested(xi) for xi in x)
    if not valid_items:
        return False

    # If all values are scalars-like items
    if all((not isinstance(xi, Sized) or isinstance(xi, Tensor0D)) for xi in x):
        return True

    # If all values are sized items with same size
    elif all(isinstance(xi, Sized) for xi in x):
        len0 = len(x[0])
        return all(len(xi) == len0 for xi in x[1:])

    else:
        return False


def __can_be_converted_to_tensor_nested(
    x: Any,
) -> bool:
    if is_builtin_number(x):
        return True
    elif isinstance(x, Tensor0D):
        return True
    elif isinstance(x, (np.ndarray, np.generic)) and x.dtype in ACCEPTED_NUMPY_DTYPES:
        return True
    elif isinstance(x, (List, Tuple)):
        return __can_be_converted_to_tensor_list_tuple(x)
    else:
        return False


def is_floating_point(x: Any) -> TypeGuard[Union[FloatingTensor, np.ndarray, float]]:
    """Returns True if object is a/contains floating-point object(s)."""
    if isinstance(x, Tensor):
        return x.is_floating_point()
    elif isinstance(x, (np.ndarray, np.generic)):
        return numpy_is_floating_point(x)
    else:
        return isinstance(x, float)


def is_complex(x: Any) -> TypeGuard[Union[ComplexFloatingTensor, np.ndarray, complex]]:
    """Returns True if object is a/contains complex-valued object(s)."""
    if isinstance(x, Tensor):
        return x.is_complex()
    elif isinstance(x, (np.ndarray, np.generic)):
        return numpy_is_complex(x)
    else:
        return isinstance(x, complex)


def is_sorted(
    x: Union[Tensor, np.ndarray, Iterable],
    *,
    reverse: bool = False,
    strict: bool = False,
) -> bool:
    """Returns True if the sequence is sorted."""
    if isinstance(x, (Tensor, np.ndarray)):
        if x.ndim != 1:
            msg = f"Invalid number of dims in argument {x.ndim=}. (expected 1)"
            raise ValueError(msg)

        prev = slice(None, -1)
        next_ = slice(1, None)

        if not reverse and not strict:
            result = x[prev] <= x[next_]
        elif not reverse and strict:
            result = x[prev] < x[next_]
        elif reverse and not strict:
            result = x[prev] >= x[next_]
        else:  # reverse and strict
            result = x[prev] > x[next_]

        result = result.all().item()
        return result  # type: ignore

    elif isinstance(x, Iterable):
        return builtin_is_sorted(x, reverse=reverse, strict=strict)

    else:
        raise TypeError(f"Invalid argument type {type(x)=}.")


@overload
def all_eq(
    x: Union[Tensor, np.ndarray, ScalarLike, Iterable],
    dim: None = None,
) -> bool:
    ...


@overload
def all_eq(
    x: T_TensorOrArray,
    dim: int,
) -> T_TensorOrArray:
    ...


def all_eq(
    x: Union[T_TensorOrArray, ScalarLike, Iterable],
    dim: Union[int, None] = None,
) -> Union[bool, T_TensorOrArray]:
    """Check if all elements are equal in a tensor, ndarray, iterable or scalar object."""
    if isinstance(x, Tensor):
        if dim is None:
            if x.ndim == 0 or x.nelement() == 0:
                return True
            x = x.reshape(-1)
            return (x[0] == x[1:]).all().item()  # type: ignore
        else:
            slices: List[Union[slice, int, None]] = [slice(None) for _ in range(x.ndim)]
            slices[dim] = 0
            slices.insert(dim + 1, None)
            return (x == x[slices]).all(dim)  # type: ignore

    elif isinstance(x, (np.ndarray, np.generic)):
        return numpy_all_eq(x, dim=dim)  # type: ignore

    elif dim is not None:
        raise ValueError(f"Invalid argument {dim=} with {type(x)=}.")

    elif is_scalar_like(x):
        return True

    elif isinstance(x, Iterable):
        return builtin_all_eq(x)

    else:
        raise TypeError(f"Invalid argument type {type(x)=}.")


def all_ne(x: Union[Tensor, np.ndarray, ScalarLike, Iterable]) -> bool:
    """Check if all elements are NOT equal in a tensor, ndarray, iterable or scalar object."""
    if isinstance(x, Tensor):
        return len(torch.unique(x)) == x.nelement()
    elif isinstance(x, (np.ndarray, np.generic)):
        return numpy_all_ne(x)
    elif is_scalar_like(x):
        return True
    elif isinstance(x, Iterable):
        return builtin_all_ne(x)
    else:
        raise TypeError(f"Invalid argument type {type(x)=}.")


def is_full(x: TensorOrArray, target: Any = ...) -> bool:
    """Check if all element are equal to target in a tensor or array. Accept an optional value 'target' to specified the expected value."""
    if nelement(x) == 0 and target is not ...:
        return False
    if nelement(x) == 0 and target is ...:
        return True

    indices = tuple([0] * x.ndim)
    first_elem = x[indices]
    return (target is ... or first_elem == target) and all_eq(x)


@function_alias(all_ne)
def is_unique(*args, **kwargs):
    ...


@deprecated_alias(is_stackable)
def can_be_stacked(*args, **kwargs):
    ...


@deprecated_alias(is_convertible_to_tensor)
def can_be_converted_to_tensor(*args, **kwargs):
    ...
