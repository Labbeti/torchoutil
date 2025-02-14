#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from dataclasses import dataclass
from numbers import Number, Integral
from typing import (
    NamedTuple,
    Any,
    Optional,
    Dict,
    Mapping,
    Iterable,
    List,
    Tuple,
    Sequence,
    Generator,
    Union,
    Callable,
    Literal,
)
from unittest import TestCase

from typing_extensions import TypeIs, TypeGuard

from torchoutil.pyoutil.typing import (
    is_dataclass_instance,
    is_iterable_str,
    is_namedtuple_instance,
    NoneType,
    NamedTupleInstance,
    BuiltinNumber,
    is_builtin_number,
    isinstance_guard,
)


class TestTypeChecks(TestCase):
    def test_is_iterable_str_1(self) -> None:
        inputs = [
            ("a", True, False),
            (["a"], True, True),
            ([], True, True),
            (("a",), True, True),
            ((), True, True),
            (1.0, False, False),
            ([["a"]], False, False),
            ("", True, False),
            ((s for s in ("a", "b", "c")), True, True),
        ]

        for x, expected_1, expected_2 in inputs:
            result_1 = is_iterable_str(x, accept_str=True)
            result_2 = is_iterable_str(x, accept_str=False)

            assert expected_1 == result_1
            assert expected_2 == result_2

    def test_is_dataclass_example_1(self) -> None:
        @dataclass
        class DC:
            a: int = 0
            b: str = "0"

        dc = DC(a=0, b="0")

        assert not is_namedtuple_instance(DC)
        assert not is_namedtuple_instance(dc)

        assert not is_dataclass_instance(DC)
        assert is_dataclass_instance(dc)

    def test_is_namedtuple_example_1(self) -> None:
        class NT1(NamedTuple):
            a: int
            b: str

        NT2 = NamedTuple("NT2", [("a", int), ("b", str)])

        nt1 = NT1(a=0, b="0")
        nt2 = NT2(a=0, b="0")

        assert not is_namedtuple_instance(NT1)
        assert not is_namedtuple_instance(NT2)
        assert is_namedtuple_instance(nt1)
        assert is_namedtuple_instance(nt2)

        assert not is_dataclass_instance(NT1)
        assert not is_dataclass_instance(NT2)
        assert not is_dataclass_instance(nt1)
        assert not is_dataclass_instance(nt2)


class TestIsInstanceGuard(TestCase):
    def test_example_1(self) -> None:
        x = 1

        assert isinstance_guard(x, int)
        assert isinstance_guard(x, Number)
        assert isinstance_guard(x, Optional[int])
        assert isinstance_guard(x, Union[int, str])
        assert isinstance_guard(x, Literal[1])
        assert isinstance_guard(x, Literal[2, None, 1, "a"])

        assert not isinstance_guard(x, float)
        assert not isinstance_guard(x, Callable)
        assert not isinstance_guard(x, Generator)
        assert not isinstance_guard(x, Literal[2])

    def test_example_2(self) -> None:
        x = {"a": 2, "b": 10}

        assert isinstance_guard(x, dict)
        assert isinstance_guard(x, Dict)
        assert isinstance_guard(x, Mapping)

        assert isinstance_guard(x, Dict[str, int])
        assert isinstance_guard(x, Dict[Any, int])
        assert isinstance_guard(x, Dict[str, Any])
        assert isinstance_guard(x, Iterable[str])
        assert isinstance_guard(x, Mapping[str, int])
        assert isinstance_guard(x, Dict[Literal["b", "a"], Literal[10, 2]])

        assert not isinstance_guard(x, set)
        assert not isinstance_guard(x, Dict[str, float])
        assert not isinstance_guard(x, Dict[Literal["a"], Literal[10, 2]])

    def test_old_compatibility(self) -> None:
        examples = [
            0,
            1.0,
            1j,
            None,
            [],
            (),
            {},
            set(),
            ("a",),
            ("a", 1, 3),
            {"a": "a"},
            [1, "a"],
        ]
        for x in examples:
            assert is_dict_str(x) == isinstance_guard(x, Dict[str, Any])
            assert is_dict_str_number(x) == isinstance_guard(x, Dict[str, Number])
            assert is_dict_str_optional_int(x) == isinstance_guard(
                x, Dict[str, Optional[int]]
            )
            assert is_dict_str_str(x) == isinstance_guard(x, Dict[str, str])

            assert is_iterable_bool(x) == isinstance_guard(x, Iterable[bool])
            assert is_iterable_bytes_or_list(x) == isinstance_guard(
                x, Iterable[Union[bytes, list]]
            )
            assert is_iterable_float(x) == isinstance_guard(x, Iterable[float])
            assert is_iterable_int(x) == isinstance_guard(x, Iterable[int])
            assert is_iterable_integral(x) == isinstance_guard(x, Iterable[Integral])
            assert is_iterable_iterable_int(x) == isinstance_guard(
                x, Iterable[Iterable[int]]
            )
            assert is_iterable_mapping_str(x) == isinstance_guard(
                x, Iterable[Mapping[str, Any]]
            )
            assert is_iterable_str(x) == isinstance_guard(x, Iterable[str])

            assert is_list_bool(x) == isinstance_guard(x, List[bool])
            assert is_list_builtin_number(x) == isinstance_guard(x, List[BuiltinNumber])
            assert is_list_float(x) == isinstance_guard(x, List[float])
            assert is_list_int(x) == isinstance_guard(x, List[int])
            assert is_list_list_str(x) == isinstance_guard(x, List[List[str]])
            assert is_list_number(x) == isinstance_guard(x, List[Number])
            assert is_list_str(x) == isinstance_guard(x, List[str])

            assert is_mapping_str(x) == isinstance_guard(x, Mapping[str, Any])
            assert is_mapping_str_iterable(x) == isinstance_guard(
                x, Mapping[str, Iterable]
            )
            assert is_namedtuple_instance(x) == isinstance_guard(x, NamedTupleInstance)

            assert is_sequence_bool(x) == isinstance_guard(x, Sequence[bool])
            assert is_sequence_int(x) == isinstance_guard(x, Sequence[int])
            assert is_sequence_str(x) == isinstance_guard(x, Sequence[str])

            assert is_tuple_optional_int(x) == isinstance_guard(
                x, Tuple[Optional[int], ...]
            )
            assert is_tuple_int(x) == isinstance_guard(x, Tuple[int, ...])
            assert is_tuple_str(x) == isinstance_guard(x, Tuple[str, ...])


def is_dict_str(x: Any) -> TypeIs[Dict[str, Any]]:
    return isinstance(x, dict) and all(isinstance(key, str) for key in x.keys())


def is_dict_str_number(x: Any) -> TypeIs[Dict[str, Number]]:
    return (
        isinstance(x, dict)
        and all(isinstance(key, str) for key in x.keys())
        and all(isinstance(value, Number) for value in x.values())
    )


def is_dict_str_optional_int(x: Any) -> TypeIs[Dict[str, Optional[int]]]:
    return (
        isinstance(x, dict)
        and all(isinstance(key, str) for key in x.keys())
        and all(isinstance(value, (int, NoneType)) for value in x.values())
    )


def is_dict_str_str(x: Any) -> TypeIs[Dict[str, str]]:
    return (
        isinstance(x, dict)
        and all(isinstance(key, str) for key in x.keys())
        and all(isinstance(value, str) for value in x.values())
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


def is_iterable_mapping_str(x: Any) -> TypeIs[Iterable[Mapping[str, Any]]]:
    return isinstance(x, Iterable) and all(
        isinstance(xi, Mapping) and all(isinstance(xik) for xik in xi.keys())
        for xi in x
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


def is_list_bool(x: Any) -> TypeIs[List[bool]]:
    return isinstance(x, list) and all(isinstance(xi, bool) for xi in x)


def is_list_builtin_number(x: Any) -> TypeIs[List[BuiltinNumber]]:
    return isinstance(x, list) and all(is_builtin_number(xi) for xi in x)


def is_list_float(x: Any) -> TypeIs[List[float]]:
    return isinstance(x, list) and (all(isinstance(xi, float) for xi in x))


def is_list_int(x: Any) -> TypeIs[List[int]]:
    return isinstance(x, list) and all(isinstance(xi, int) for xi in x)


def is_list_list_str(x: Any) -> TypeIs[List[List[str]]]:
    return (
        isinstance(x, list)
        and all(isinstance(xi, list) for xi in x)
        and all(isinstance(xij, str) for xi in x for xij in xi)
    )


def is_list_number(x: Any) -> TypeIs[List[Number]]:
    return isinstance(x, list) and all(isinstance(xi, Number) for xi in x)


def is_list_str(x: Any) -> TypeIs[List[str]]:
    return isinstance(x, list) and all(isinstance(xi, str) for xi in x)


def is_mapping_str(x: Any) -> TypeIs[Mapping[str, Any]]:
    return isinstance(x, Mapping) and all(isinstance(key, str) for key in x.keys())


def is_mapping_str_iterable(x: Any) -> TypeIs[Mapping[str, Iterable[Any]]]:
    return is_mapping_str(x) and all(isinstance(v, Iterable) for v in x.values())


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


if __name__ == "__main__":
    unittest.main()
