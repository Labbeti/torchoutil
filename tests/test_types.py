#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

import torch

from torchoutil.types import (
    is_builtin_number,
    is_number_like,
    is_numpy_number_like,
    is_tensor0d,
    np,
)
from torchoutil.utils.packaging import _NUMPY_AVAILABLE


class TestIsNumber(TestCase):
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
            result = is_number_like(x)
            msg = f"{x=} ({is_builtin_number(x)}, {is_tensor0d(x)}, {is_numpy_number_like(x)})"
            assert result == expected, msg


if __name__ == "__main__":
    unittest.main()
