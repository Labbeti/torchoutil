#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.others import (
    can_be_converted_to_tensor,
    can_be_stacked,
    is_numpy_scalar,
    is_python_scalar,
    is_scalar,
    is_torch_scalar,
)
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np


class TestCanBeConvertedToTensor(TestCase):
    def test_examples(self) -> None:
        examples = [
            [[1, 0, 0], [2, 3, 4]],
            [[1, 0, 0], [2, 3]],
            [[[True], [False]], [[False], [True]]],
            [[[]], [[]]],
            [torch.rand(10), torch.rand(10)],
            [torch.rand(10, 5), torch.rand(10, 3)],
            [torch.rand(10, 5), torch.rand(10, 5)],
            [[torch.rand(10)]],
            torch.rand(10, 5),
            [(), []],
        ]

        if _NUMPY_AVAILABLE:
            examples += [
                np.float64(42),
                [[np.float64(42)], np.ndarray([2])],
                np.random.rand(10, 5),
                [[np.float128(42)], np.ndarray([2])],
            ]

        for example in examples:
            try:
                torch.as_tensor(example)
                convertible = True
            except ValueError:
                convertible = False
            except TypeError:
                convertible = False

            try:
                torch.stack(example)
                stackable = True
            except TypeError:
                stackable = False
            except RuntimeError:
                stackable = False

            assert (
                can_be_converted_to_tensor(example) == convertible
            ), f"can_be_converted_to_tensor: {example=}"
            assert can_be_stacked(example) == stackable, f"can_be_stacked: {example=}"


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
