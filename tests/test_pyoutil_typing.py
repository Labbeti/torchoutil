#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from dataclasses import dataclass
from typing import NamedTuple
from unittest import TestCase

from torchoutil.pyoutil import (
    is_dataclass_instance,
    is_iterable_str,
    is_namedtuple_instance,
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


if __name__ == "__main__":
    unittest.main()
