#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import math
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Tensor, nn

# backward compatibility
from torchoutil.core.get import get_device, get_dtype, get_generator  # noqa: F401
from torchoutil.extras.numpy import np
from torchoutil.nn import functional as F
from torchoutil.pyoutil.collections import all_eq as builtin_all_eq
from torchoutil.pyoutil.collections import prod as builtin_prod
from torchoutil.pyoutil.collections import unzip
from torchoutil.pyoutil.functools import function_alias, identity
from torchoutil.pyoutil.semver import Version
from torchoutil.pyoutil.typing import BuiltinNumber, SizedIter, T_BuiltinNumber
from torchoutil.types._typing import (
    LongTensor,
    ScalarLike,
    T_TensorOrArray,
    TensorOrArray,
)
from torchoutil.types.guards import is_scalar_like
from torchoutil.types.tensor_subclasses import Tensor0D, Tensor1D, Tensor2D, Tensor3D
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
    value: Any,
    x: Tensor,
    *,
    default: Union[None, Tensor, BuiltinNumber] = None,
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
def get_ndim(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    return_valid: Literal[False] = False,
) -> int:
    ...


@overload
def get_ndim(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    return_valid: Literal[True],
) -> return_types.ndim:
    ...


def get_ndim(
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


@function_alias(get_ndim)
def ndim(*args, **kwargs):
    ...


@overload
def get_shape(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    output_type: Callable[[Tuple[int, ...]], T] = identity,
    return_valid: Literal[False] = False,
) -> T:
    ...


@overload
def get_shape(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    output_type: Callable[[Tuple[int, ...]], T] = identity,
    return_valid: Literal[True],
) -> return_types.shape[T]:
    ...


def get_shape(
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


@function_alias(get_shape)
def shape(*args, **kwargs):
    ...


def ranks(x: Tensor, dim: int = -1, descending: bool = False) -> LongTensor:
    """Get the ranks of each value in range [0, x.shape[dim][."""
    return x.argsort(dim, descending).argsort(dim)  # type: ignore


def nelement(x: Union[ScalarLike, Tensor, np.ndarray, Iterable]) -> int:
    """Returns the number of elements in Tensor-like object."""
    if isinstance(x, Tensor):
        return x.nelement()
    elif isinstance(x, (np.ndarray, np.generic)):
        return x.size
    else:
        return builtin_prod(get_shape(x))


@overload
def prod(
    x: T_TensorOrArray,
    *,
    dim: Optional[int] = None,
    start: Any = 1,
) -> T_TensorOrArray:
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
    x: Union[TensorOrArray, Iterable[T_BuiltinNumber]],
    *,
    dim: Optional[int] = None,
    start: T_BuiltinNumber = 1,
) -> Union[Tensor, T_BuiltinNumber]:
    """Returns the product of all elements in input."""
    if isinstance(x, Tensor):
        if dim is not None:
            return torch.prod(x, dim=dim) * start
        else:
            return torch.prod(x) * start
    elif isinstance(x, np.ndarray):
        return np.prod(x, axis=dim, initial=start)
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


def mse(
    x1: Tensor,
    x2: Tensor,
    *,
    dim: Union[int, Tuple[int, ...], None] = None,
) -> Tensor:
    """Mean squared error function."""
    if dim is not None or Version(torch.__version__) >= "2.0.0":
        return ((x1 - x2) ** 2).mean(dim).sqrt()  # type: ignore
    else:
        return ((x1 - x2) ** 2).mean().sqrt()


def rmse(
    x1: Tensor,
    x2: Tensor,
    *,
    dim: Union[int, Tuple[int, ...], None] = None,
) -> Tensor:
    """Root mean squared error function."""
    return mse(x1, x2, dim=dim).sqrt()


def deep_equal(x: T, y: T) -> bool:
    if is_scalar_like(x) and is_scalar_like(y):
        x_isnan = math.isnan(x) if F.is_floating_point(x) else False
        y_isnan = math.isnan(y) if F.is_floating_point(y) else False
        return (x_isnan and y_isnan) or F.to_item(x == y)  # type: ignore

    if isinstance(x, Tensor) and isinstance(y, Tensor):
        x_isnan = x.isnan()
        y_isnan = y.isnan()
        return (
            (x.shape == y.shape)
            and bool((x_isnan == y_isnan).all().item())
            and torch.equal(x[~x_isnan], y[~y_isnan])
        )

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        x_isnan = (
            np.isnan(x)
            if F.is_floating_point(x)
            else np.full(x.shape, False, dtype=bool)
        )
        y_isnan = (
            np.isnan(y)
            if F.is_floating_point(y)
            else np.full(y.shape, False, dtype=bool)
        )
        return (
            (x.shape == y.shape)
            and (x_isnan == y_isnan).all().item()
            and np.equal(x[~x_isnan], y[~y_isnan]).all().item()
        )

    if isinstance(x, Mapping) and isinstance(y, Mapping):
        return deep_equal(list(x.items()), list(y.items()))
    if isinstance(x, SizedIter) and isinstance(y, SizedIter):
        return len(x) == len(y) and all(deep_equal(xi, yi) for xi, yi in zip(x, y))

    return (type(x) is type(y)) and (x == y)  # type: ignore


@overload
def stack(
    tensors: Union[List[Tensor0D], Tuple[Tensor0D, ...]],
    dim: int = 0,
    *,
    out: Optional[Tensor1D] = None,
) -> Tensor1D:
    ...


@overload
def stack(
    tensors: Union[List[Tensor1D], Tuple[Tensor1D, ...]],
    dim: int = 0,
    *,
    out: Optional[Tensor2D] = None,
) -> Tensor2D:
    ...


@overload
def stack(
    tensors: Union[List[Tensor2D], Tuple[Tensor2D, ...]],
    dim: int = 0,
    *,
    out: Optional[Tensor3D] = None,
) -> Tensor3D:
    ...


@overload
def stack(
    tensors: Union[List[Tensor], Tuple[Tensor, ...]],
    dim: int = 0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    ...


def stack(
    tensors,
    dim: int = 0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:  # type: ignore
    return torch.stack(tensors, dim=dim, out=out)


def cat(
    tensors: Union[List[Tensor], Tuple[Tensor, ...]],
    dim: int = 0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    return torch.cat(tensors, dim=dim, out=out)


@function_alias(cat)
def concat(*args, **kwargs):
    ...
