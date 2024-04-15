#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from torch import Tensor, nn
from typing_extensions import TypeGuard

from torchoutil.nn.functional.get import get_device

T = TypeVar("T")
U = TypeVar("U")


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    params = (p for p in model.parameters() if not only_trainable or p.requires_grad)
    count = sum(p.numel() for p in params)
    return count


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


def is_python_scalar(x: Any) -> TypeGuard[Union[int, float, bool, complex]]:
    return isinstance(x, (int, float, bool, complex))


def is_torch_scalar(x: Any) -> TypeGuard[Tensor]:
    return isinstance(x, Tensor) and x.ndim == 0


def is_scalar(x: Any) -> TypeGuard[Union[int, float, bool, complex, Tensor]]:
    return is_python_scalar(x) or is_torch_scalar(x)


def can_be_stacked(
    tensors: Union[List[Any], Tuple[Any, ...]],
) -> TypeGuard[Union[List[Tensor], Tuple[Tensor, ...]]]:
    if len(tensors) == 0:
        return True
    if not all(isinstance(tensor, Tensor) for tensor in tensors):
        return False
    shape0 = tensors[0].shape
    result = all(tensor.shape == shape0 for tensor in tensors[1:])
    return result


def can_be_converted_to_tensor(x: Any) -> bool:
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

    if all(not isinstance(xi, Sized) for xi in x):
        return True
    elif all(isinstance(xi, Sized) for xi in x):
        len0 = len(x[0])
        return all(len(xi) == len0 for xi in x[1:])
    else:
        return False


def __can_be_converted_to_tensor_nested(x: Any) -> bool:
    if is_python_scalar(x):
        return True
    elif isinstance(x, (List, Tuple)):
        return __can_be_converted_to_tensor_list_tuple(x)
    else:
        return False
