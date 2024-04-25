#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.others import (
    can_be_converted_to_tensor,
    is_numpy_scalar,
    is_python_scalar,
    is_scalar,
    is_torch_scalar,
)
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np


class TestCanBeConvertedToTensor(TestCase):
    def test_example_1(self) -> None:
        lst = [[1, 0, 0], [2, 3, 4]]
        self.assertTrue(can_be_converted_to_tensor(lst))

    def test_example_2(self) -> None:
        lst = [[1, 0, 0], [2, 3]]
        self.assertFalse(can_be_converted_to_tensor(lst))

    def test_example_3(self) -> None:
        lst = [[[True], [False]], [[False], [True]]]
        self.assertTrue(can_be_converted_to_tensor(lst))

    def test_example_4(self) -> None:
        lst = [[[]], [[]]]
        self.assertTrue(can_be_converted_to_tensor(lst))

    def test_example_5(self) -> None:
        lst = [torch.rand(10), torch.rand(10)]
        self.assertFalse(can_be_converted_to_tensor(lst))

    def test_example_6(self) -> None:
        lst = [[torch.rand(10)]]
        self.assertFalse(can_be_converted_to_tensor(lst))

    def test_example_7(self) -> None:
        lst = torch.rand(10)
        self.assertTrue(can_be_converted_to_tensor(lst))

    def test_example_8(self) -> None:
        lst = [(), []]
        self.assertTrue(can_be_converted_to_tensor(lst))


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
