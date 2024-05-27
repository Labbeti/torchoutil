#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from dataclasses import dataclass
from typing import NamedTuple
from unittest import TestCase

import numpy as np
import torch

from torchoutil.nn.functional.numpy import _NUMPY_AVAILABLE
from torchoutil.utils.type_checks import (
    is_dataclass_instance,
    is_iterable_str,
    is_namedtuple_instance,
    is_numpy_scalar,
    is_python_scalar,
    is_scalar,
    is_torch_scalar,
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


class TestIsScalar(TestCase):
    def test_example_1(self) -> None:
        tests = [
            (1, True),
            (1.0, True),
            (False, True),
            (1j + 2, True),
            (torch.rand(()), True),
            (torch.rand((0,)), False),
            (torch.rand(10), False),
            ([], False),
            ([[]], False),
            ((), False),
            ("test", False),
        ]

        if _NUMPY_AVAILABLE:
            tests += [
                (np.float64(random.random()), True),
                (np.int64(random.randint(0, 100)), True),
                (np.random.random(()), True),
                (np.random.random((0,)), False),
                (np.random.random((10,)), False),
            ]

        for x, expected in tests:
            result = is_scalar(x)
            msg = f"{x=} ({is_python_scalar(x)}, {is_torch_scalar(x)}, {is_numpy_scalar(x)})"
            assert result == expected, msg


if __name__ == "__main__":
    unittest.main()
