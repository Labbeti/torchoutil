#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Tensor, nn

from torchoutil.extras.numpy import np
from torchoutil.pyoutil.collections import all_eq as builtin_all_eq
from torchoutil.pyoutil.collections import prod as builtin_prod
from torchoutil.pyoutil.collections import unzip
from torchoutil.pyoutil.functools import identity
from torchoutil.pyoutil.typing import T_BuiltinNumber
from torchoutil.types._typing import LongTensor, ScalarLike, T_TensorOrArray
from torchoutil.types.guards import is_scalar_like
from torchoutil.utils import return_types

T = TypeVar("T")
U = TypeVar("U")


def count_parameters(
    model: nn.Module,
    *,
    recurse: bool = True,
    only_trainable: bool = False,
    buffers: bool = False,
) -> int:
    """Returns the number of parameters in a module."""
    params = (
        param
        for param in model.parameters(recurse)
        if not only_trainable or param.requires_grad
    )

    if buffers:
        params = itertools.chain(params, (buffer for buffer in model.buffers(recurse)))

    num_params = sum(param.numel() for param in params)
    return num_params


def find(
    x: Tensor,
    value: Any,
    default: Union[None, Tensor, int, float] = None,
    dim: int = -1,
) -> LongTensor:
    """Return the index of the first occurrence of value in a tensor."""
    if x.ndim == 0:
        msg = f"Function 'find' does not supports 0-d tensors. (found {x.ndim=})"
        raise ValueError(msg)

    mask = x.eq(value)
    contains = mask.any(dim=dim)
    indices = mask.long().argmax(dim=dim)

    if default is None:
        if not contains.all():
            raise RuntimeError(f"Cannot find {value=} in tensor.")
        return indices  # type: ignore
    else:
        output = torch.where(contains, indices, default)
        return output  # type: ignore


@overload
def ndim(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    return_valid: Literal[False] = False,
) -> int:
    ...


@overload
def ndim(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    return_valid: Literal[True],
) -> return_types.ndim:
    ...


def ndim(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    return_valid: bool = False,
    use_first_for_list_tuple: bool = False,
) -> Union[int, return_types.ndim]:
    """Scan first argument to return its number of dimension(s). Works recursively with Tensors, numpy arrays and builtins types instances.

    Note: Sets and dicts are considered as scalars with a shape equal to 0.

    Args:
        x: Input value to scan.
        return_valid: If True, returns a tuple containing a boolean indicator if the data has an homogeneous ndim instead of raising a ValueError. defaults to False.
        use_first_for_list_tuple: If True, use first value to determine ndim for list and tuple argument. Otherwise it will scan each value in argument to determine its shape. defaults to False.

    Raises:
        ValueError if input has an heterogeneous number of dimensions.
        TypeError if input has an unsupported type.
    """

    def _impl(
        x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    ) -> Tuple[bool, int]:
        if is_scalar_like(x):
            return True, 0
        elif isinstance(x, (Tensor, np.ndarray, np.generic)):
            return True, x.ndim
        elif isinstance(x, (set, frozenset, dict)):
            return True, 0
        elif isinstance(x, (list, tuple)):
            valids_and_ndims = unzip(_impl(xi) for xi in x)  # type: ignore
            if len(valids_and_ndims) == 0:
                return True, 1

            valids, ndims = valids_and_ndims
            if (use_first_for_list_tuple and valids[0]) or (
                all(valids) and builtin_all_eq(ndims)
            ):
                return True, ndims[0] + 1
            else:
                return False, -1
        else:
            raise TypeError(f"Invalid argument type {type(x)}.")

    valid, ndim = _impl(x)
    if return_valid:
        return return_types.ndim(valid, ndim)
    elif valid:
        return ndim
    else:
        msg = f"Invalid argument {x}. (cannot compute ndim for heterogeneous data)"
        raise ValueError(msg)


@overload
def shape(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    output_type: Callable[[Tuple[int, ...]], T] = identity,
    return_valid: Literal[False] = False,
) -> T:
    ...


@overload
def shape(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    output_type: Callable[[Tuple[int, ...]], T] = identity,
    return_valid: Literal[True],
) -> return_types.shape[T]:
    ...


def shape(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    output_type: Callable[[Tuple[int, ...]], T] = identity,
    return_valid: bool = False,
    use_first_for_list_tuple: bool = False,
) -> Union[T, return_types.shape[T]]:
    """Scan first argument to return its shape. Works recursively with Tensors, numpy arrays and builtins types instances.

    Note: Sets and dicts are considered as scalars with a shape equal to ().

    Args:
        x: Input value to scan.
        output_type: Output shape type. defaults to identity, which returns a tuple of ints.
        return_valid: If True, returns a tuple containing a boolean indicator if the data has an homogeneous shape instead of raising a ValueError. defaults to False.
        use_first_for_list_tuple: If True, use first value to determine ndim for list and tuple argument. Otherwise it will scan each value in argument to determine its shape. defaults to False.

    Raises:
        ValueError: if input has an heterogeneous shape.
        TypeError: if input has an unsupported type.
    """

    def _impl(
        x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    ) -> Tuple[bool, Tuple[int, ...]]:
        if is_scalar_like(x):
            return True, ()
        elif isinstance(x, (Tensor, np.ndarray, np.generic)):
            return True, tuple(x.shape)
        elif isinstance(x, (set, frozenset, dict)):
            return True, ()
        elif isinstance(x, (list, tuple)):
            valids_and_shapes = unzip(_impl(xi) for xi in x)  # type: ignore
            if len(valids_and_shapes) == 0:
                return True, (0,)

            valids, shapes = valids_and_shapes
            if (use_first_for_list_tuple and valids[0]) or (
                all(valids) and builtin_all_eq(shapes)
            ):
                return True, (len(shapes),) + shapes[0]
            else:
                return False, ()
        else:
            raise TypeError(f"Invalid argument type {type(x)}.")

    valid, shape = _impl(x)
    if return_valid:
        shape = output_type(shape)
        return return_types.shape(valid, shape)
    elif valid:
        shape = output_type(shape)
        return shape
    else:
        msg = f"Invalid argument {x}. (cannot compute shape for heterogeneous data)"
        raise ValueError(msg)


def ranks(x: Tensor, dim: int = -1, descending: bool = False) -> Tensor:
    """Get the ranks of each value in range [0, x.shape[dim][."""
    return x.argsort(dim, descending).argsort(dim)


def nelement(x: Union[ScalarLike, Tensor, np.ndarray, Iterable]) -> int:
    """Returns the number of elements in Tensor-like object."""
    if isinstance(x, Tensor):
        return x.nelement()
    elif isinstance(x, (np.ndarray, np.generic)):
        return x.size
    else:
        return builtin_prod(shape(x))


@overload
def prod(
    x: Tensor,
    *,
    dim: Optional[int] = None,
    start: Any = 1,
) -> Tensor:
    ...


@overload
def prod(
    x: Iterable[T_BuiltinNumber],
    *,
    dim: Any = None,
    start: T_BuiltinNumber = 1,
) -> T_BuiltinNumber:
    ...


def prod(
    x: Union[Tensor, Iterable[T_BuiltinNumber]],
    *,
    dim: Optional[int] = None,
    start: T_BuiltinNumber = 1,
) -> Union[Tensor, T_BuiltinNumber]:
    """Returns the product of all elements in input."""
    if isinstance(x, Tensor):
        return torch.prod(x, dim=dim)
    elif isinstance(x, np.ndarray):
        return np.prod(x, axis=dim)
    elif isinstance(x, Iterable):
        if dim is not None:
            msg = f"Invalid argument {dim=}. (expected None with {type(x)=})"
            raise ValueError(msg)
        return builtin_prod(x, start=start)  # type: ignore
    else:
        msg = (
            f"Invalid argument type {type(x)=}. (expected Tensor, ndarray or Iterable)"
        )
        raise TypeError(msg)


def average_power(
    x: T_TensorOrArray,
    dim: Union[int, Tuple[int, ...], None] = -1,
) -> T_TensorOrArray:
    """Compute average power of a signal along a specified dim/axis."""
    return (abs(x) ** 2).mean(dim)  # type: ignore
