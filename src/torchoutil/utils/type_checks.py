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
    Union,
    runtime_checkable,
)

from torch import Tensor
from typing_extensions import TypeGuard


@runtime_checkable
class DataclassInstance(Protocol):
    # Class meant for typing purpose only
    __dataclass_fields__: ClassVar[Dict[str, Any]]


@runtime_checkable
class NamedTupleInstance(Protocol):
    # Class meant for typing purpose only
    _fields: tuple[str, ...]
    _fields_defaults: Dict[str, Any]

    def _asdict(self) -> Dict[str, Any]:
        ...

    def __getitem__(self, idx):
        ...

    def __len__(self) -> int:
        ...


def is_dataclass_instance(x: Any) -> TypeGuard[DataclassInstance]:
    return isinstance(x, DataclassInstance)


def is_namedtuple_instance(x: Any) -> TypeGuard[NamedTupleInstance]:
    return isinstance(x, NamedTupleInstance)


def is_dict_str(x: Any) -> TypeGuard[Dict[str, Any]]:
    return isinstance(x, dict) and all(isinstance(key, str) for key in x.keys())


def is_iterable_int(x: Any) -> TypeGuard[Iterable[int]]:
    return isinstance(x, Iterable) and all(isinstance(xi, int) for xi in x)


def is_iterable_str(x: Any, *, accept_str: bool) -> TypeGuard[Iterable[str]]:
    return (accept_str and isinstance(x, str)) or (
        not isinstance(x, str)
        and isinstance(x, Iterable)
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


def is_list_str(x: Any) -> TypeGuard[List[str]]:
    return isinstance(x, list) and all(isinstance(xi, str) for xi in x)


def is_list_tensor(x: Any) -> TypeGuard[List[Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x)


def is_mapping_str(x: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(x, Mapping) and all(isinstance(key, str) for key in x.keys())
