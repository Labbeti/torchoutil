#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from numbers import Integral, Number
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import typing_extensions
from typing_extensions import (
    NotRequired,
    Required,
    TypeGuard,
    TypeIs,
    TypeVar,
    get_args,
    get_origin,
)

from ..warnings import deprecated_function
from .classes import (
    BuiltinNumber,
    BuiltinScalar,
    DataclassInstance,
    NamedTupleInstance,
    NoneType,
)

T = TypeVar("T")

pylog = logging.getLogger(__name__)


def is_typed_dict(x: Any) -> TypeGuard[type]:
    if sys.version_info.major == 3 and sys.version_info.minor < 9:
        return x.__class__.__name__ == "_TypedDictMeta"
    else:
        return hasattr(x, "__orig_bases__") and TypedDict in x.__orig_bases__


def isinstance_guard(
    obj: Any,
    class_or_tuple: Union[Type[T], None, Tuple[Type[T], ...]],
) -> TypeIs[T]:
    """Improved isinstance(...) function that supports parametrized Union, TypedDict, Literal, Mapping or Iterable.

    Example 1::
    -----------
    ```
    >>> isinstance_guard({"a": 1, "b": 2}, dict[str, int])  # True
    >>> isinstance_guard({"a": 1, "b": 2}, dict[str, str])  # False
    >>> isinstance_guard({"a": 1, "b": 2}, dict)  # True
    ```
    """
    if isinstance(obj, type):
        return False
    if class_or_tuple is Any or class_or_tuple is typing_extensions.Any:
        return True
    if class_or_tuple is None:
        return obj is None
    if isinstance(class_or_tuple, tuple):
        return any(
            isinstance_guard(obj, target_type_i) for target_type_i in class_or_tuple
        )
    if is_typed_dict(class_or_tuple):
        return _isinstance_guard_typed_dict(obj, class_or_tuple)

    origin = get_origin(class_or_tuple)
    if origin is None:
        return isinstance(obj, class_or_tuple)

    # Special case for empty tuple because get_args(Tuple[()]) returns () and not ((),) in python >= 3.11
    # More info at https://github.com/python/cpython/issues/91137
    if class_or_tuple == Tuple[()]:
        return obj == ()

    args = get_args(class_or_tuple)
    if len(args) == 0:
        return isinstance_guard(obj, origin)

    if origin is Union:
        return any(isinstance_guard(obj, arg) for arg in args)

    if origin is Literal:
        return obj in args

    if isinstance(obj, Generator):
        msg = f"Invalid argument type {type(obj)}."
        raise TypeError(msg)

    if issubclass(origin, Mapping):
        assert len(args) == 2, f"{args=}"
        if not isinstance_guard(obj, origin):
            return False

        return all(isinstance_guard(k, args[0]) for k in obj.keys()) and all(
            isinstance_guard(v, args[1]) for v in obj.values()
        )

    if issubclass(origin, Tuple):
        if not isinstance_guard(obj, origin):
            return False
        elif len(args) == 1 and args[0] == ():
            return len(obj) == 0
        elif len(args) == 2 and args[1] is ...:
            args = tuple([args[0]] * len(obj))
        elif len(obj) != len(args):
            return False
        return all(isinstance_guard(xi, ti) for xi, ti in zip(obj, args))

    if issubclass(origin, Iterable):
        if not isinstance_guard(obj, origin):
            return False
        return all(isinstance_guard(xi, args[0]) for xi in obj)

    msg = f"Unsupported type {class_or_tuple}. (expected unparametrized type or parametrized Union, TypedDict, Literal, Mapping or Iterable)"
    raise NotImplementedError(msg)


def _isinstance_guard_typed_dict(x: Any, target_type: type) -> bool:
    if not isinstance_guard(x, Dict[str, Any]):
        return False

    total: bool = target_type.__total__
    annotations = target_type.__annotations__

    required_annotations = {}
    optional_annotations = {}
    for k, v in annotations.items():
        origin = get_origin(v)
        if origin is Required:
            required_annotations[k] = v
        elif origin is NotRequired:
            optional_annotations[k] = v
        elif total:
            required_annotations[k] = v
        else:
            optional_annotations[k] = v

    if not set(required_annotations.keys()).issubset(x.keys()):
        return False
    if not (
        set(required_annotations.keys()) | set(optional_annotations.keys())
    ).issuperset(x.keys()):
        return False

    for k, v in required_annotations.items():
        origin = get_origin(v)
        if origin is Required:
            v = get_args(v)[0]

        if not isinstance_guard(x[k], v):
            return False

    for k, v in optional_annotations.items():
        if k not in x:
            continue
        origin = get_origin(v)
        if origin is NotRequired:
            v = get_args(v)[0]
        if not isinstance_guard(x[k], v):
            return False

    return True


def is_builtin_obj(x: Any) -> bool:
    """Returns True if object is an instance of a builtin object.

    Note: If the object is an instance of a custom subtype of a builtin object, this function returns False.
    """
    return x.__class__.__module__ == "builtins" and not isinstance(x, type)


def is_builtin_number(x: Any, *, strict: bool = False) -> TypeIs[BuiltinNumber]:
    """Returns True if x is an instance of a builtin number type (int, float, bool, complex).

    Args:
        x: Object to check.
        strict: If True, it will not consider custom subtypes of builtins as builtin numbers. defaults to False.
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


def is_dataclass_instance(x: Any) -> TypeIs[DataclassInstance]:
    """Returns True if argument is a dataclass.

    Unlike function `dataclasses.is_dataclass`, this function returns False for a dataclass type.
    """
    return isinstance_guard(x, DataclassInstance)


def is_namedtuple_instance(x: Any) -> TypeIs[NamedTupleInstance]:
    """Returns True if argument is a NamedTuple."""
    return isinstance_guard(x, NamedTupleInstance)


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
    return isinstance_guard(x, Iterable[int]) and (
        accept_bool or not isinstance_guard(x, Iterable[bool])
    )


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
@deprecated_function("{fn_name}, use `isinstance_guard(x, Dict[str, Any])` instead.")
def is_dict_str(x: Any) -> TypeIs[Dict[str, Any]]:
    return isinstance_guard(x, Dict[str, Any])


@deprecated_function(
    "{fn_name}, use `isinstance_guard(x, Dict[str, Optional[int]])` instead."
)
def is_dict_str_optional_int(x: Any) -> TypeIs[Dict[str, Optional[int]]]:
    return isinstance_guard(x, Dict[str, Optional[int]])


@deprecated_function("{fn_name}, use `isinstance_guard(x, Dict[str, Number])` instead.")
def is_dict_str_number(x: Any) -> TypeIs[Dict[str, Number]]:
    return isinstance_guard(x, Dict[str, Number])


@deprecated_function("{fn_name}, use `isinstance_guard(x, Dict[str, str])` instead.")
def is_dict_str_str(x: Any) -> TypeIs[Dict[str, str]]:
    return isinstance_guard(x, Dict[str, str])


@deprecated_function(
    "{fn_name}, use `isinstance_guard(x, Iterable[Iterable[int]])` instead."
)
def is_iterable_iterable_int(x: Any) -> TypeIs[Iterable[Iterable[int]]]:
    return isinstance_guard(x, Iterable[Iterable[int]])


@deprecated_function(
    "{fn_name}, use `isinstance_guard(x, Iterable[Mapping[str, Any]])` instead."
)
def is_iterable_mapping_str(x: Any) -> TypeIs[Iterable[Mapping[str, Any]]]:
    return isinstance_guard(x, Iterable[Mapping[str, Any]])


@deprecated_function("{fn_name}, use `isinstance_guard(x, List[bool])` instead.")
def is_list_bool(x: Any) -> TypeIs[List[bool]]:
    return isinstance_guard(x, List[bool])


@deprecated_function(
    "{fn_name}, use `isinstance_guard(x, List[BuiltinNumber])` instead."
)
def is_list_builtin_number(x: Any) -> TypeIs[List[BuiltinNumber]]:
    return isinstance_guard(x, List[BuiltinNumber])


@deprecated_function("{fn_name}, use `isinstance_guard(x, List[float])` instead.")
def is_list_float(x: Any) -> TypeIs[List[float]]:
    return isinstance_guard(x, List[float])


@deprecated_function("{fn_name}, use `isinstance_guard(x, List[int])` instead.")
def is_list_int(x: Any) -> TypeIs[List[int]]:
    return isinstance_guard(x, List[int])


@deprecated_function("{fn_name}, use `isinstance_guard(x, List[List[str]])` instead.")
def is_list_list_str(x: Any) -> TypeIs[List[List[str]]]:
    return isinstance_guard(x, List[List[str]])


@deprecated_function("{fn_name}, use `isinstance_guard(x, List[Number])` instead.")
def is_list_number(x: Any) -> TypeIs[List[Number]]:
    return isinstance_guard(x, List[Number])


@deprecated_function("{fn_name}, use `isinstance_guard(x, List[str])` instead.")
def is_list_str(x: Any) -> TypeIs[List[str]]:
    return isinstance_guard(x, List[str])


@deprecated_function("{fn_name}, use `isinstance_guard(x, Mapping[str, Any])` instead.")
def is_mapping_str(x: Any) -> TypeIs[Mapping[str, Any]]:
    return isinstance_guard(x, Mapping[str, Any])


@deprecated_function(
    "{fn_name}, use `isinstance_guard(x, Mapping[str, Iterable[Any]])` instead."
)
def is_mapping_str_iterable(x: Any) -> TypeIs[Mapping[str, Iterable[Any]]]:
    return isinstance_guard(x, Mapping[str, Iterable[Any]])


@deprecated_function("{fn_name}, use `isinstance_guard(x, Sequence[bool])` instead.")
def is_sequence_bool(x: Any) -> TypeIs[Sequence[bool]]:
    return isinstance_guard(x, Sequence[bool])


@deprecated_function("{fn_name}, use `isinstance_guard(x, Sequence[float])` instead.")
def is_sequence_float(x: Any) -> TypeIs[Sequence[float]]:
    return isinstance_guard(x, Sequence[float])


@deprecated_function("{fn_name}, use `isinstance_guard(x, Sequence[int])` instead.")
def is_sequence_int(x: Any) -> TypeIs[Sequence[int]]:
    return isinstance_guard(x, Sequence[int])


@deprecated_function(
    "{fn_name}, use `isinstance_guard(x, Tuple[Optional[int], ...])` instead."
)
def is_tuple_optional_int(x: Any) -> TypeIs[Tuple[Optional[int], ...]]:
    return isinstance_guard(x, Tuple[Optional[int], ...])


@deprecated_function("{fn_name}, use `isinstance_guard(x, Tuple[int, ...])` instead.")
def is_tuple_int(x: Any) -> TypeIs[Tuple[int, ...]]:
    return isinstance_guard(x, Tuple[int, ...])


@deprecated_function("{fn_name}, use `isinstance_guard(x, Tuple[str, ...])` instead.")
def is_tuple_str(x: Any) -> TypeIs[Tuple[str, ...]]:
    return isinstance_guard(x, Tuple[str, ...])
