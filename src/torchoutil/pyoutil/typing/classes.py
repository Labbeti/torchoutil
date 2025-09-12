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
    Union,
    runtime_checkable,
)

from typing_extensions import TypeAlias, TypeVar

NoneType: TypeAlias = type(None)  # type: ignore
EllipsisType: TypeAlias = type(...)  # type: ignore

BuiltinCollection: TypeAlias = Union[list, tuple, dict, set, frozenset]
BuiltinNumber: TypeAlias = Union[bool, int, float, complex]
BuiltinScalar: TypeAlias = Union[bool, int, float, complex, NoneType, str, bytes]

T = TypeVar("T", covariant=True)
T_BuiltinNumber = TypeVar("T_BuiltinNumber", bound=BuiltinNumber, default=BuiltinNumber)
T_BuiltinScalar = TypeVar("T_BuiltinScalar", bound=BuiltinScalar, default=BuiltinScalar)


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
        raise NotImplementedError

    def __getitem__(self, idx, /):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@runtime_checkable
class SupportsIterLen(Protocol[T]):
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@runtime_checkable
class SupportsGetitemLen(Protocol[T]):
    def __getitem__(self, idx, /) -> T:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@runtime_checkable
class SupportsGetitemIterLen(Protocol[T]):
    def __getitem__(self, idx, /) -> T:
        raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@runtime_checkable
class SupportsBool(Protocol):
    def __bool__(self) -> bool:
        raise NotImplementedError


@runtime_checkable
class SupportsAdd(Protocol):
    def __add__(self, other, /):
        raise NotImplementedError


@runtime_checkable
class SupportsAnd(Protocol):
    def __and__(self, other, /):
        raise NotImplementedError


@runtime_checkable
class SupportsMul(Protocol):
    def __mul__(self, other, /):
        raise NotImplementedError


@runtime_checkable
class SupportsOr(Protocol):
    def __or__(self, other, /):
        raise NotImplementedError


# Aliases for backward compatibility
SizedIter = SupportsIterLen
SizedIterable = SupportsIterLen

SizedGetitem = SupportsGetitemLen
SupportsLenAndGetItem = SupportsGetitemLen

SizedGetitemIter = SupportsGetitemIterLen
SupportsLenAndGetItemAndIter = SupportsGetitemIterLen

BoolLike = Union[bool, int, SupportsBool, Sized]
