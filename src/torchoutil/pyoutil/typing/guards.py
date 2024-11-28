#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numbers import Integral, Number
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from typing_extensions import TypeGuard, TypeIs

from .classes import (
    BuiltinNumber,
    BuiltinScalar,
    DataclassInstance,
    NamedTupleInstance,
    NoneType,
)


def is_builtin_obj(x: Any) -> bool:
    """Returns True if object is an instance of a builtin object.

    Note: If the object is an instance of a custom subtype of a builtin object, this function returns False.
    """
    return x.__class__.__module__ == "builtins" and not isinstance(x, type)


def is_builtin_number(x: Any, *, strict: bool = False) -> TypeIs[BuiltinNumber]:
    """Returns True if x is an instance of a builtin number type (int, float, bool, complex).

    Args:
        x: Object to check.
        strict: If True, it will not consider subtypes of builtins as builtin numbers.
    """
    if strict and not is_builtin_obj(x):
        return False
    return isinstance(x, (int, float, bool, complex))


def is_builtin_scalar(x: Any, *, strict: bool = False) -> TypeIs[BuiltinScalar]:
    """Returns True if x is an instance of a builtin scalar type (int, float, bool, complex, NoneType, str, bytes).

    Args:
        x: Object to check.
        strict: If True, it will not consider subtypes of builtins as builtin numbers.
    """
    if strict and not is_builtin_obj(x):
        return False
    return isinstance(x, (int, float, bool, complex, NoneType, str, bytes))


def is_dataclass_instance(x: Any) -> TypeIs[DataclassInstance]:
    """Returns True if argument is a dataclass. Unlike function `dataclasses.is_dataclass`, this function returns False for a dataclass type."""
    return not isinstance(x, type) and isinstance(x, DataclassInstance)


def is_dict_str(x: Any) -> TypeIs[Dict[str, Any]]:
    return isinstance(x, dict) and all(isinstance(key, str) for key in x.keys())


def is_dict_str_optional_int(x: Any) -> TypeIs[Dict[str, Optional[int]]]:
    return (
        isinstance(x, dict)
        and all(isinstance(key, str) for key in x.keys())
        and all(isinstance(value, (int, NoneType)) for value in x.values())
    )


def is_iterable_bool(
    x: Any,
    *,
    accept_generator: bool = True,
) -> TypeIs[Iterable[bool]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance(x, Iterable) and all(isinstance(xi, bool) for xi in x)


def is_iterable_bytes_or_list(
    x: Any,
    *,
    accept_generator: bool = True,
) -> TypeIs[Iterable[Union[bytes, list]]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance(x, Iterable) and all(isinstance(xi, (bytes, list)) for xi in x)


def is_iterable_float(
    x: Any,
    *,
    accept_generator: bool = True,
) -> TypeIs[Iterable[float]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance(x, Iterable) and all(isinstance(xi, float) for xi in x)


def is_iterable_int(
    x: Any,
    *,
    accept_bool: bool = True,
    accept_generator: bool = True,
) -> TypeIs[Iterable[int]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance(x, Iterable) and all(
        isinstance(xi, int) and (accept_bool or not isinstance(xi, bool)) for xi in x
    )


def is_iterable_integral(
    x: Any,
    *,
    accept_generator: bool = True,
) -> TypeIs[Iterable[Integral]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance(x, Iterable) and all(isinstance(xi, Integral) for xi in x)


def is_iterable_iterable_int(x: Any) -> TypeIs[Iterable[Iterable[int]]]:
    return (
        isinstance(x, Iterable)
        and all(isinstance(xi, Iterable) for xi in x)
        and all(isinstance(xij, int) for xi in x for xij in xi)
    )


def is_iterable_str(
    x: Any,
    *,
    accept_str: bool = True,
    accept_generator: bool = True,
) -> TypeGuard[Iterable[str]]:
    if isinstance(x, str):
        return accept_str
    if isinstance(x, Generator):
        return accept_generator and all(isinstance(xi, str) for xi in x)
    return isinstance(x, Iterable) and all(isinstance(xi, str) for xi in x)


def is_list_list_str(x: Any) -> TypeIs[List[List[str]]]:
    return (
        isinstance(x, list)
        and all(isinstance(xi, list) for xi in x)
        and all(isinstance(xij, str) for xi in x for xij in xi)
    )


def is_list_bool(x: Any) -> TypeIs[List[bool]]:
    return isinstance(x, list) and all(isinstance(xi, bool) for xi in x)


def is_list_float(x: Any) -> TypeIs[List[float]]:
    return isinstance(x, list) and (all(isinstance(xi, float) for xi in x))


def is_list_int(x: Any) -> TypeIs[List[int]]:
    return isinstance(x, list) and all(isinstance(xi, int) for xi in x)


def is_list_number(x: Any) -> TypeIs[List[Number]]:
    return isinstance(x, list) and all(isinstance(xi, Number) for xi in x)


def is_list_builtin_number(x: Any) -> TypeIs[List[BuiltinNumber]]:
    return isinstance(x, list) and all(is_builtin_number(xi) for xi in x)


def is_list_str(x: Any) -> TypeIs[List[str]]:
    return isinstance(x, list) and all(isinstance(xi, str) for xi in x)


def is_mapping_str(x: Any) -> TypeIs[Mapping[str, Any]]:
    return isinstance(x, Mapping) and all(isinstance(key, str) for key in x.keys())


def is_namedtuple_instance(x: Any) -> TypeIs[NamedTupleInstance]:
    return not isinstance(x, type) and isinstance(x, NamedTupleInstance)


def is_sequence_bool(x: Any) -> TypeIs[Sequence[bool]]:
    return isinstance(x, Sequence) and all(isinstance(xi, bool) for xi in x)


def is_sequence_int(x: Any) -> TypeIs[Sequence[int]]:
    return isinstance(x, Sequence) and all(isinstance(xi, int) for xi in x)


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


def is_tuple_optional_int(x: Any) -> TypeIs[Tuple[Optional[int], ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, (int, NoneType)) for xi in x)


def is_tuple_int(x: Any) -> TypeIs[Tuple[int, ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, int) for xi in x)


def is_tuple_str(x: Any) -> TypeIs[Tuple[str, ...]]:
    return isinstance(x, tuple) and all(isinstance(xi, str) for xi in x)
