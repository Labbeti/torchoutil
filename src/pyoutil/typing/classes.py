#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    ClassVar,
    Dict,
    Protocol,
    Sized,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

NoneType = type(None)
EllipsisType = type(...)

BuiltinCollection = Union[list, tuple, dict, set, frozenset]
BuiltinNumber = Union[int, float, bool, complex]
BuiltinScalar = Union[int, float, bool, complex, str, bytes, NoneType]

T = TypeVar("T", covariant=True)
TBuiltinScalar = TypeVar("TBuiltinScalar", bound=BuiltinScalar)


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


@runtime_checkable
class SizedIterable(Protocol[T]):
    def __iter__(self) -> T:
        ...

    def __len__(self) -> int:
        ...


@runtime_checkable
class SupportsLenAndGetItem(Protocol[T]):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx) -> T:
        ...


@runtime_checkable
class SupportsBool(Protocol):
    def __bool__(self) -> bool:
        ...


BoolLike = Union[SupportsBool, Sized]
