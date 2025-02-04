#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
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

from typing_extensions import (
    Any,
    TypeGuard,
    Type,
    TypeIs,
    TypeVar,
    Mapping,
    Iterable,
    get_args,
    get_origin,
)
from typing_extensions import TypeGuard, TypeIs

from .classes import (
    BuiltinNumber,
    BuiltinScalar,
    DataclassInstance,
    NamedTupleInstance,
    NoneType,
)

T = TypeVar("T")


def isinstance_guard(x: Any, target_type: Type[T]) -> TypeIs[T]:
    if isinstance(x, type):
        return False
    if target_type is Any:
        return True

    origin = get_origin(target_type)
    if origin is None:
        return isinstance(x, target_type)

    args = get_args(target_type)
    if origin is Union:
        return any(isinstance_guard(x, arg) for arg in args)

    if issubclass(origin, Mapping):
        assert len(args) in (0, 2), f"{args=}"
        if not isinstance_guard(x, origin):
            return False
        if len(args) == 0:
            return True
        return all(isinstance_guard(k, args[0]) for k in x.keys()) and all(
            isinstance_guard(v, args[1]) for v in x.values()
        )

    if issubclass(origin, Iterable):
        if not isinstance_guard(x, origin):
            return False
        if len(args) == 0:
            return True
        return all(isinstance_guard(xi, args[0]) for xi in x)

    raise NotImplementedError(
        f"Unsupported type {target_type}. (expected unparametrized type, Union, Mapping or Iterable)"
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
        strict: If True, it will not consider subtypes of builtins as builtin numbers. defaults to False.
    """
    if strict and not is_builtin_obj(x):
        return False
    return isinstance(x, (int, float, bool, complex, NoneType, str, bytes))


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
    return isinstance_guard(x, Iterable[str])


def is_iterable_bool(
    x: Any,
    *,
    accept_generator: bool = True,
) -> TypeIs[Iterable[bool]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance_guard(x, Iterable[bool])


def is_iterable_bytes_or_list(
    x: Any,
    *,
    accept_generator: bool = True,
) -> TypeIs[Iterable[Union[bytes, list]]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance_guard(x, Iterable[Union[bytes, list]])


def is_iterable_float(
    x: Any,
    *,
    accept_generator: bool = True,
) -> TypeIs[Iterable[float]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance_guard(x, Iterable[float])


def is_iterable_int(
    x: Any,
    *,
    accept_bool: bool = True,
    accept_generator: bool = True,
) -> TypeIs[Iterable[int]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance_guard(x, Iterable[int])


def is_iterable_integral(
    x: Any,
    *,
    accept_generator: bool = True,
) -> TypeIs[Iterable[Integral]]:
    if not accept_generator and isinstance(x, Generator):
        return False
    return isinstance_guard(x, Iterable[Integral])


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


# -----------------------
# Old aliases
# -----------------------
def is_dataclass_instance(x: Any) -> TypeIs[DataclassInstance]:
    """Returns True if argument is a dataclass. Unlike function `dataclasses.is_dataclass`, this function returns False for a dataclass type."""
    return isinstance_guard(x, DataclassInstance)


def is_dict_str(x: Any) -> TypeIs[Dict[str, Any]]:
    return isinstance_guard(x, Dict[str, Any])


def is_dict_str_optional_int(x: Any) -> TypeIs[Dict[str, Optional[int]]]:
    return isinstance_guard(x, Dict[str, Optional[int]])


def is_dict_str_number(x: Any) -> TypeIs[Dict[str, Number]]:
    return isinstance_guard(x, Dict[str, Number])


def is_dict_str_str(x: Any) -> TypeIs[Dict[str, str]]:
    return isinstance_guard(x, Dict[str, str])


def is_iterable_iterable_int(x: Any) -> TypeIs[Iterable[Iterable[int]]]:
    return isinstance_guard(x, Iterable[Iterable[int]])


def is_iterable_mapping_str(x: Any) -> TypeIs[Iterable[Mapping[str, Any]]]:
    return isinstance_guard(x, Iterable[Mapping[str, Any]])


def is_list_bool(x: Any) -> TypeIs[List[bool]]:
    return isinstance_guard(x, List[bool])


def is_list_builtin_number(x: Any) -> TypeIs[List[BuiltinNumber]]:
    return isinstance_guard(x, List[BuiltinNumber])


def is_list_float(x: Any) -> TypeIs[List[float]]:
    return isinstance_guard(x, List[float])


def is_list_int(x: Any) -> TypeIs[List[int]]:
    return isinstance_guard(x, List[int])


def is_list_list_str(x: Any) -> TypeIs[List[List[str]]]:
    return isinstance_guard(x, List[List[str]])


def is_list_number(x: Any) -> TypeIs[List[Number]]:
    return isinstance_guard(x, List[Number])


def is_list_str(x: Any) -> TypeIs[List[str]]:
    return isinstance_guard(x, List[str])


def is_mapping_str(x: Any) -> TypeIs[Mapping[str, Any]]:
    return isinstance_guard(x, Mapping[str, Any])


def is_mapping_str_iterable(x: Any) -> TypeIs[Mapping[str, Iterable[Any]]]:
    return isinstance_guard(x, Mapping[str, Iterable])


def is_namedtuple_instance(x: Any) -> TypeIs[NamedTupleInstance]:
    return isinstance_guard(x, NamedTupleInstance)


def is_sequence_bool(x: Any) -> TypeIs[Sequence[bool]]:
    return isinstance_guard(x, Sequence[bool])


def is_sequence_int(x: Any) -> TypeIs[Sequence[float]]:
    return isinstance_guard(x, Sequence[float])


def is_sequence_int(x: Any) -> TypeIs[Sequence[int]]:
    return isinstance_guard(x, Sequence[int])


def is_tuple_optional_int(x: Any) -> TypeIs[Tuple[Optional[int], ...]]:
    return isinstance_guard(x, Tuple[Optional[int], ...])


def is_tuple_int(x: Any) -> TypeIs[Tuple[int, ...]]:
    return isinstance_guard(x, Tuple[int, ...])


def is_tuple_str(x: Any) -> TypeIs[Tuple[str, ...]]:
    return isinstance_guard(x, Tuple[str, ...])
