#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numbers import Number
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from typing_extensions import TypeGuard

from .classes import BuiltinNumber, BuiltinScalar, DataclassInstance, NamedTupleInstance


def is_builtin_number(x: Any) -> TypeGuard[BuiltinNumber]:
    """Returns True if x is a builtin scalar type (int, float, bool, complex)."""
    return isinstance(x, BuiltinNumber)


def is_builtin_scalar(x: Any) -> TypeGuard[BuiltinScalar]:
    """Returns True if x is a builtin scalar type (int, float, bool, complex)."""
    return isinstance(x, BuiltinScalar)


def is_dataclass_instance(x: Any) -> TypeGuard[DataclassInstance]:
    return not isinstance(x, type) and isinstance(x, DataclassInstance)


def is_dict_str(x: Any) -> TypeGuard[Dict[str, Any]]:
    return isinstance(x, dict) and all(isinstance(key, str) for key in x.keys())


def is_iterable_bool(x: Any) -> TypeGuard[Iterable[bool]]:
    return isinstance(x, Iterable) and (all(isinstance(xi, bool) for xi in x))


def is_iterable_float(x: Any) -> TypeGuard[Iterable[float]]:
    return isinstance(x, Iterable) and (all(isinstance(xi, float) for xi in x))


def is_iterable_bytes_or_list(x: Any) -> TypeGuard[Iterable[Union[bytes, list]]]:
    return isinstance(x, Iterable) and all(isinstance(xi, (bytes, list)) for xi in x)


def is_iterable_int(x: Any) -> TypeGuard[Iterable[int]]:
    return isinstance(x, Iterable) and (all(isinstance(xi, int) for xi in x))


def is_iterable_iterable_int(x: Any) -> TypeGuard[Iterable[Iterable[int]]]:
    return (
        isinstance(x, Iterable)
        and all(isinstance(xi, Iterable) for xi in x)
        and all(isinstance(xij, int) for xi in x for xij in xi)
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


def is_mapping_str(x: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(x, Mapping) and all(isinstance(key, str) for key in x.keys())


def is_list_list_str(x: Any) -> TypeGuard[List[List[str]]]:
    return (
        isinstance(x, list)
        and all(isinstance(xi, list) for xi in x)
        and all(isinstance(xij, str) for xi in x for xij in xi)
    )


def is_list_bool(x: Any) -> TypeGuard[List[bool]]:
    return isinstance(x, list) and all(isinstance(xi, bool) for xi in x)


def is_list_float(x: Any) -> TypeGuard[List[float]]:
    return isinstance(x, list) and (all(isinstance(xi, float) for xi in x))


def is_list_int(x: Any) -> TypeGuard[List[int]]:
    return isinstance(x, list) and all(isinstance(xi, int) for xi in x)


def is_list_number(x: Any) -> TypeGuard[List[Number]]:
    return isinstance(x, list) and all(isinstance(xi, Number) for xi in x)


def is_list_str(x: Any) -> TypeGuard[List[str]]:
    return isinstance(x, list) and all(isinstance(xi, str) for xi in x)


def is_namedtuple_instance(x: Any) -> TypeGuard[NamedTupleInstance]:
    return not isinstance(x, type) and isinstance(x, NamedTupleInstance)


def is_sequence_bool(x: Any) -> TypeGuard[Sequence[bool]]:
    return isinstance(x, Sequence) and (all(isinstance(xi, bool) for xi in x))


def is_sequence_int(x: Any) -> TypeGuard[Sequence[int]]:
    return isinstance(x, Sequence) and (all(isinstance(xi, int) for xi in x))


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


def is_tuple_str(x: Any) -> TypeGuard[Tuple[str, ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, str) for xi in x)
