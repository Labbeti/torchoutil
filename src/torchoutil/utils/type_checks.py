#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import torch
from torch import Tensor
from typing_extensions import TypeGuard

from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np


@runtime_checkable
class DataclassInstance(Protocol):
    # Class meant for typing purpose only
    __dataclass_fields__: ClassVar[Dict[str, Any]]


@runtime_checkable
class NamedTupleInstance(Protocol):
    # Class meant for typing purpose only
    _fields: Tuple[str, ...]
    _field_defaults: Dict[str, Any]

    def _asdict(self) -> Dict[str, Any]:
        ...

    def __getitem__(self, idx):
        ...

    def __len__(self) -> int:
        ...


BuiltinScalar = Union[int, float, bool, complex]


def is_numpy_scalar(x: Any) -> bool:
    """Returns True if x is an instance of a numpy generic type or a zero-dimensional numpy array.
    If numpy is not installed, this function always returns False.
    """
    if _NUMPY_AVAILABLE:
        return isinstance(x, np.generic) or (isinstance(x, np.ndarray) and x.ndim == 0)
    else:
        return False


def is_python_scalar(x: Any) -> TypeGuard[Union[BuiltinScalar]]:
    """Returns True if x is a builtin scalar type (int, float, bool, complex)."""
    return isinstance(x, (int, float, bool, complex))


def is_torch_scalar(x: Any) -> TypeGuard[Tensor]:
    """Returns True if x is a zero-dimensional torch Tensor."""
    return isinstance(x, Tensor) and x.ndim == 0


def is_scalar(x: Any) -> TypeGuard[Union[BuiltinScalar, Tensor]]:
    """Returns True if input is a scalar.

    Accepted scalars list is:
    - Python numbers (int, float, bool, complex)
    - PyTorch zero-dimensional tensors
    - Numpy zero-dimensional arrays
    - Numpy generic scalars
    """
    return (
        is_python_scalar(x)
        or is_torch_scalar(x)
        or (_NUMPY_AVAILABLE and is_numpy_scalar(x))
    )


def is_dataclass_instance(x: Any) -> TypeGuard[DataclassInstance]:
    return not isinstance(x, type) and isinstance(x, DataclassInstance)


def is_namedtuple_instance(x: Any) -> TypeGuard[NamedTupleInstance]:
    return not isinstance(x, type) and isinstance(x, NamedTupleInstance)


def is_dict_str(x: Any) -> TypeGuard[Dict[str, Any]]:
    return isinstance(x, dict) and all(isinstance(key, str) for key in x.keys())


def is_iterable_bool(
    x: Any,
    *,
    accept_tensor: bool = False,
) -> TypeGuard[Iterable[bool]]:
    return isinstance(x, Iterable) and (
        all(isinstance(xi, bool) for xi in x)
        or (
            accept_tensor
            and all(
                isinstance(xi, Tensor) and xi.ndim == 0 and xi.dtype == torch.bool
                for xi in x
            )
        )
    )


def is_sequence_bool(
    x: Any,
    *,
    accept_tensor: bool = False,
) -> TypeGuard[Sequence[bool]]:
    return isinstance(x, Sequence) and (
        all(isinstance(xi, bool) for xi in x)
        or (
            accept_tensor
            and all(
                isinstance(xi, Tensor) and xi.ndim == 0 and xi.dtype == torch.bool
                for xi in x
            )
        )
    )


def is_sequence_int(
    x: Any,
    *,
    accept_tensor: bool = False,
) -> TypeGuard[Sequence[int]]:
    return isinstance(x, Sequence) and (
        all(isinstance(xi, int) for xi in x)
        or (
            accept_tensor
            and all(
                isinstance(xi, Tensor)
                and xi.ndim == 0
                and not xi.is_floating_point()
                and not xi.is_complex()
                for xi in x
            )
        )
    )


def is_iterable_int(
    x: Any,
    *,
    accept_tensor: bool = False,
) -> TypeGuard[Iterable[int]]:
    return isinstance(x, Iterable) and (
        all(isinstance(xi, int) for xi in x)
        or (
            accept_tensor
            and all(
                isinstance(xi, Tensor)
                and xi.ndim == 0
                and not xi.is_floating_point()
                and not xi.is_complex()
                for xi in x
            )
        )
    )


def is_iterable_str(
    x: Any,
    *,
    accept_str: bool = True,
) -> TypeGuard[Iterable[str]]:
    return (accept_str and isinstance(x, str)) or (
        not isinstance(x, str)
        and isinstance(x, Iterable)
        and all(isinstance(xi, str) for xi in x)
    )


def is_sequence_str(
    x: Any,
    *,
    accept_str: bool = True,
) -> TypeGuard[Sequence[str]]:
    return (accept_str and isinstance(x, str)) or (
        not isinstance(x, str)
        and isinstance(x, Sequence)
        and all(isinstance(xi, str) for xi in x)
    )


def is_iterable_bytes_list(x: Any) -> TypeGuard[Iterable[Union[bytes, list]]]:
    return isinstance(x, Iterable) and all(isinstance(xi, (bytes, list)) for xi in x)


def is_iterable_iterable_int(x: Any) -> TypeGuard[Iterable[Iterable[int]]]:
    return (
        isinstance(x, Iterable)
        and all(isinstance(xi, Iterable) for xi in x)
        and all(isinstance(xij, int) for xi in x for xij in xi)
    )


def is_iterable_tensor(x: Any) -> TypeGuard[Iterable[Tensor]]:
    return isinstance(x, Iterable) and all(isinstance(xi, Tensor) for xi in x)


def is_list_list_str(x: Any) -> TypeGuard[List[List[str]]]:
    return (
        isinstance(x, list)
        and all(isinstance(xi, list) for xi in x)
        and all(isinstance(xij, str) for xi in x for xij in xi)
    )


def is_list_bool(x: Any) -> TypeGuard[List[bool]]:
    return isinstance(x, list) and all(isinstance(xi, bool) for xi in x)


def is_list_int(x: Any) -> TypeGuard[List[int]]:
    return isinstance(x, list) and all(isinstance(xi, int) for xi in x)


def is_list_str(x: Any) -> TypeGuard[List[str]]:
    return isinstance(x, list) and all(isinstance(xi, str) for xi in x)


def is_list_tensor(x: Any) -> TypeGuard[List[Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x)


def is_mapping_str(x: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(x, Mapping) and all(isinstance(key, str) for key in x.keys())


def is_tuple_tensor(x: Any) -> TypeGuard[Tuple[Tensor, ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, Tensor) for xi in x)


def is_tuple_str(x: Any) -> TypeGuard[Tuple[str, ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, str) for xi in x)
