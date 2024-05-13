#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.others import can_be_converted_to_tensor, can_be_stacked
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
                [[np.complex128(42)], np.ndarray([2])],
                [np.float16(42), np.float32(99)],
            ]

        for example in examples:
            try:
                torch.as_tensor(example)
                convertible = True
            except TypeError:
                convertible = False
            except ValueError:
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


if __name__ == "__main__":
    unittest.main()
