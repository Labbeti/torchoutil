#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest
from typing import Iterable, Literal, Mapping
from unittest import TestCase

from torchoutil.pyoutil import inspect
from torchoutil.pyoutil.inspect import get_fullname


class DummyClass:
    def f(self):
        return 0


class TestPyoutilInspect(TestCase):
    def test_example_1(self) -> None:
        x = [0, 1, 2]
        assert get_fullname(x) == "builtins.list(...)"

        x = 1.0
        assert get_fullname(x) == "builtins.float(...)"

        x = DummyClass()
        assert get_fullname(x) == f"{self.__module__}.DummyClass(...)"
        assert get_fullname(x.f) == f"{self.__module__}.DummyClass.f"

        assert get_fullname(DummyClass) == f"{self.__module__}.DummyClass"
        assert get_fullname(DummyClass.f) == f"{self.__module__}.DummyClass.f"

    def test_example_2(self) -> None:
        assert get_fullname(TestCase) == "unittest.case.TestCase"
        assert get_fullname(inspect) == "torchoutil.pyoutil.inspect"
        assert get_fullname(get_fullname) == "torchoutil.pyoutil.inspect.get_fullname"

        if sys.version_info.minor >= 11:
            assert get_fullname(Mapping) == "typing.Mapping"
            assert get_fullname(Iterable[str]) == "typing.Iterable[builtins.str]"
            assert (
                get_fullname(Iterable[Literal[1]])
                == "typing.Iterable[typing.Literal[builtins.int(...)]]"
            )


if __name__ == "__main__":
    unittest.main()
