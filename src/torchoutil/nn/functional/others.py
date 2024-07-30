#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import struct
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
from torchoutil.types import (
    ACCEPTED_NUMPY_DTYPES,
    is_builtin_scalar,
    is_list_tensor,
    is_numpy_scalar,
    is_scalar,
    is_torch_scalar,
    is_tuple_tensor,
    np,
)
from torchoutil.utils.stdlib.collections import all_eq

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

    if isinstance(x, (str, float, int, bool, complex)):
        return x
    elif isinstance(x, (Tensor, nn.Module)):
        if predicate is None or predicate(x):
            return x.to(**kwargs)
        else:
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
    """Returns True if inputs can be passed to `torch.stack` function, i.e. contanis a list or tuple of tensors with the same shape.

    Alias of :func:`~torchoutil.nn.functional.others.is_stackable`.
    """
    return is_stackable(tensors)


def is_stackable(
    tensors: Union[List[Any], Tuple[Any, ...]],
) -> TypeGuard[Union[List[Tensor], Tuple[Tensor, ...]]]:
    """Returns True if inputs can be passed to `torch.stack` function, i.e. contanis a list or tuple of tensors with the same shape."""
    if not is_list_tensor(tensors) and not is_tuple_tensor(tensors):
        return False
    if len(tensors) == 0:
        return False
    shape0 = tensors[0].shape
    result = all(tensor.shape == shape0 for tensor in tensors[1:])
    return result


def can_be_converted_to_tensor(x: Any) -> bool:
    """Returns True if inputs can be passed to `torch.as_tensor` function.

    Alias of :func:`~torchoutil.nn.functional.others.is_convertible_to_tensor`.

    This function returns False for heterogeneous inputs like `[[], 1]`, but this kind of value can be accepted by `torch.as_tensor`.
    """
    return is_convertible_to_tensor(x)


def is_convertible_to_tensor(x: Any) -> bool:
    """Returns True if inputs can be passed to `torch.as_tensor` function.

    This function returns False for heterogeneous inputs like `[[], 1]`, but this kind of value can be accepted by `torch.as_tensor`.
    """
    if isinstance(x, Tensor):
        return True
    else:
        return __can_be_converted_to_tensor_nested(x)


def ndim(x: Any) -> int:
    """Scan first argument to return its number of dimension(s). Works with Tensors, numpy arrays and builtins types instances."""
    valid, ndim = _search_ndim(x)
    if valid:
        return ndim
    else:
        raise ValueError(
            f"Invalid argument {x}. (cannot compute ndim for heterogeneous data)"
        )


def shape(x: Any, *, output_type: Callable[[Tuple[int, ...]], T] = Size) -> T:
    """Scan first argument to return its shape. Works with Tensors, numpy arrays and builtins types instances."""
    valid, shape = _search_shape(x)
    if valid:
        shape = output_type(shape)
        return shape
    else:
        raise ValueError(
            f"Invalid argument {x}. (cannot compute shape for heterogeneous data)"
        )


def item(x: Any) -> Union[int, float, bool, complex]:
    if is_builtin_scalar(x):
        return x
    elif is_torch_scalar(x) or is_numpy_scalar(x):
        return x.item()
    else:
        raise ValueError(f"Invalid argument {x=}. (expected scalar number object)")


def _search_ndim(x: Any) -> Tuple[bool, int]:
    if is_scalar(x) or isinstance(x, str):
        return True, 0
    elif isinstance(x, Tensor) or isinstance(x, np.ndarray):
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
    if is_scalar(x) or isinstance(x, str) or x is None:
        return True, ()
    elif isinstance(x, Tensor) or isinstance(x, np.ndarray):
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
    if is_builtin_scalar(x):
        return True
    elif isinstance(x, Tensor) and x.ndim == 0:
        return True
    elif isinstance(x, (np.ndarray, np.generic)) and x.dtype in ACCEPTED_NUMPY_DTYPES:
        return True
    elif isinstance(x, (List, Tuple)):
        return __can_be_converted_to_tensor_list_tuple(x)
    else:
        return False


def identity(x: T) -> T:
    """Identity function placeholder."""
    return x


def ranks(x: Tensor, dim: int = -1, descending: bool = False) -> Tensor:
    """Get the ranks of each value in range [0, x.shape[dim][."""
    return x.argsort(dim, descending).argsort(dim)


def checksum_module(m: nn.Module, *, only_trainable: bool = False) -> int:
    """Compute a simple checksum over module parameters."""
    return checksum_tensor(
        torch.as_tensor(
            [
                checksum_tensor(p)
                for p in m.parameters()
                if not only_trainable or p.requires_grad
            ]
        )
    )


def checksum_tensor(x: Tensor) -> int:
    """Compute a simple checksum of a tensor. Order of values matter for the checksum."""
    if x.ndim > 0:
        x = x.detach().flatten().cpu()

        if x.dtype == torch.bool:
            dtype = torch.int
        elif x.is_complex():
            dtype = x.real.dtype
        else:
            dtype = x.dtype

        x = x * torch.arange(1, len(x) + 1, device=x.device, dtype=dtype)
        x = x.nansum()

    x = x.item()
    x = _checksum_number(x)
    return x


def _checksum_number(x: Union[int, bool, complex, float]) -> int:
    """Compute a simple checksum of a builtin scalar number."""
    if isinstance(x, bool):
        return _checksum_bool(x)
    elif isinstance(x, int):
        return _checksum_int(x)
    elif isinstance(x, complex):
        return _checksum_complex(x)
    elif isinstance(x, float):
        return _checksum_float(x)
    else:
        raise TypeError(
            f"Invalid argument type {type(x)}. (expected int, bool, complex or float)"
        )


def _checksum_int(x: int) -> int:
    return x


def _checksum_bool(x: bool) -> int:
    return int(x)


def _checksum_complex(x: complex) -> int:
    return checksum_tensor(torch.as_tensor([x.real, x.imag]))


def _checksum_float(x: float) -> int:
    x = struct.pack("!f", x)
    x = struct.unpack("!i", x)[0]
    return x
