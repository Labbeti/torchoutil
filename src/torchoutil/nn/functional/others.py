#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sized,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Size, Tensor, nn
from typing_extensions import TypeGuard

from torchoutil.nn.functional.get import get_device
from torchoutil.nn.functional.numpy import ACCEPTED_NUMPY_DTYPES
from torchoutil.utils.collections import all_eq
from torchoutil.utils.packaging import _NUMPY_AVAILABLE
from torchoutil.utils.type_checks import (
    is_list_tensor,
    is_numpy_scalar,
    is_python_scalar,
    is_scalar,
    is_torch_scalar,
    is_tuple_tensor,
)

if _NUMPY_AVAILABLE:
    import numpy as np


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
) -> Tensor:
    """Return the index of the first occurrence of value in a tensor."""
    if x.ndim == 0:
        raise ValueError(
            f"Function find does not supports 0-d tensors. (found {x.ndim=} == 0)"
        )
    mask = x.eq(value)
    contains = mask.any(dim=dim)
    indices = mask.int().argmax(dim=dim)

    if default is None:
        if not contains.all():
            raise RuntimeError(f"Cannot find {value=} in tensor.")
        return indices
    else:
        output = torch.where(contains, indices, default)
        return output


@overload
def move_to_rec(
    x: Mapping[T, U],
    predicate: Optional[Callable[[Union[Tensor, nn.Module]], bool]] = None,
    **kwargs,
) -> Dict[T, U]:
    ...


@overload
def move_to_rec(
    x: T,
    predicate: Optional[Callable[[Union[Tensor, nn.Module]], bool]] = None,
    **kwargs,
) -> T:
    ...


def move_to_rec(
    x: Any,
    predicate: Optional[Callable[[Union[Tensor, nn.Module]], bool]] = None,
    **kwargs,
) -> Any:
    """Move all modules and tensors recursively to a specific dtype or device."""
    if "device" in kwargs:
        kwargs["device"] = get_device(kwargs["device"])

    if isinstance(x, (Tensor, nn.Module)):
        if predicate is None or predicate(x):
            return x.to(**kwargs)
        else:
            return x
    elif isinstance(x, (str, float, int, bool, complex)):
        return x
    elif isinstance(x, Mapping):
        return {k: move_to_rec(v, predicate=predicate, **kwargs) for k, v in x.items()}
    elif isinstance(x, Iterable):
        generator = (move_to_rec(xi, predicate=predicate, **kwargs) for xi in x)
        if isinstance(x, Generator):
            return generator
        elif isinstance(x, tuple):
            return tuple(generator)
        else:
            return list(generator)
    else:
        return x


def can_be_stacked(
    tensors: Union[List[Any], Tuple[Any, ...]],
) -> TypeGuard[Union[List[Tensor], Tuple[Tensor, ...]]]:
    """Returns True if inputs can be passed to `torch.stack` function."""
    if not is_list_tensor(tensors) and not is_tuple_tensor(tensors):
        return False
    if len(tensors) == 0:
        return False
    shape0 = tensors[0].shape
    result = all(tensor.shape == shape0 for tensor in tensors[1:])
    return result


def can_be_converted_to_tensor(x: Any) -> bool:
    """Returns True if inputs can be passed to `torch.as_tensor` function.

    This function returns False for heterogeneous inputs like `[[], 1]`, but this kind of value can be accepted by `torch.as_tensor`.
    """
    if isinstance(x, Tensor):
        return True
    else:
        return __can_be_converted_to_tensor_nested(x)


def ndim(x: Any) -> int:
    valid, ndim = _search_ndim(x)
    if valid:
        return ndim
    else:
        raise ValueError(f"Invalid argument {x}. (cannot compute ndim)")


def shape(x: Any) -> Size:
    valid, shape = _search_shape(x)
    if valid:
        shape = Size(shape)
        return shape
    else:
        raise ValueError(f"Invalid argument {x}. (cannot compute shape)")


def item(x: Any) -> Union[int, float, bool, complex]:
    if is_python_scalar(x):
        return x
    elif is_torch_scalar(x) or is_numpy_scalar(x):
        return x.item()
    else:
        raise ValueError(f"Invalid argument {x=}. (expected scalar number object)")


def _search_ndim(x: Any) -> Tuple[bool, int]:
    if is_scalar(x) or isinstance(x, str):
        return True, 0
    elif isinstance(x, Tensor) or (_NUMPY_AVAILABLE and isinstance(x, np.ndarray)):
        return True, x.ndim
    elif isinstance(x, Iterable):
        ndims = [_search_ndim(xi)[1] for xi in x]
        if len(ndims) == 0:
            return True, 1
        elif all_eq(ndims):
            return True, ndims[0] + 1
        else:
            return False, -1
    else:
        raise TypeError(f"Invalid argument type {type(x)}.")


def _search_shape(x: Any) -> Tuple[bool, Tuple[int, ...]]:
    if is_scalar(x) or isinstance(x, str):
        return True, ()
    elif isinstance(x, Tensor) or (_NUMPY_AVAILABLE and isinstance(x, np.ndarray)):
        return True, tuple(x.shape)
    elif isinstance(x, Iterable):
        shapes = [_search_shape(xi)[1] for xi in x]
        if len(shapes) == 0:
            return True, (0,)
        elif all_eq(shapes):
            return True, (len(shapes),) + shapes[0]
        else:
            return False, ()
    else:
        raise TypeError(f"Invalid argument type {type(x)}.")


def __can_be_converted_to_tensor_list_tuple(x: Union[List, Tuple]) -> bool:
    if len(x) == 0:
        return True

    valid_items = all(__can_be_converted_to_tensor_nested(xi) for xi in x)
    if not valid_items:
        return False

    if all(
        (not isinstance(xi, Sized) or (isinstance(xi, Tensor) and xi.ndim == 0))
        for xi in x
    ):
        return True
    elif all(isinstance(xi, Sized) for xi in x):
        len0 = len(x[0])
        return all(len(xi) == len0 for xi in x[1:])
    else:
        return False


def __can_be_converted_to_tensor_nested(x: Any) -> bool:
    if is_python_scalar(x):
        return True
    elif isinstance(x, Tensor) and x.ndim == 0:
        return True
    elif (
        _NUMPY_AVAILABLE
        and isinstance(x, (np.ndarray, np.generic))
        and x.dtype in ACCEPTED_NUMPY_DTYPES
    ):
        return True
    elif isinstance(x, (List, Tuple)):
        return __can_be_converted_to_tensor_list_tuple(x)
    else:
        return False
