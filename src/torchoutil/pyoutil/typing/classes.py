#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
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
BuiltinNumber = Union[bool, int, float, complex]
BuiltinScalar = Union[bool, int, float, complex, NoneType, str, bytes]

T = TypeVar("T", covariant=True)
T_BuiltinNumber = TypeVar("T_BuiltinNumber", bound=BuiltinNumber)
T_BuiltinScalar = TypeVar("T_BuiltinScalar", bound=BuiltinScalar)


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

    def __getitem__(self, idx, /):
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
    def __getitem__(self, idx, /) -> T:
        ...

    def __len__(self) -> int:
        ...


@runtime_checkable
class SupportsLenAndGetItemAndIter(Protocol[T]):
    def __getitem__(self, idx, /) -> T:
        ...

    def __iter__(self) -> Iterator[T]:
        ...

    def __len__(self) -> int:
        ...


@runtime_checkable
class SupportsBool(Protocol):
    def __bool__(self) -> bool:
        ...


BoolLike = Union[bool, int, SupportsBool, Sized]
