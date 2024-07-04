#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

T = TypeVar("T", covariant=True)
BuiltinScalar = Union[int, float, bool, complex]


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
class SizedIterable(Generic[T], Protocol):
    def __iter__(self) -> T:
        ...

    def __len__(self) -> int:
        ...
