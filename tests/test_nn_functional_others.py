#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.others import (
    can_be_converted_to_tensor,
    can_be_stacked,
    ndim,
    shape,
)
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np


class TestCanBeConvertedToTensor(TestCase):
    def test_examples(self) -> None:
        examples = [
            [],
            (),
            [[1, 0, 0], [2, 3, 4]],
            [[1, 0, 0], [2, 3]],
            [[[True], [False]], [[False], [True]]],
            [[[]], [[]]],
            [torch.rand(10), torch.rand(10)],
            [torch.rand(10, 5), torch.rand(10, 3)],
            [torch.rand(10, 5), torch.rand(10, 5)],
            [torch.rand(10, 5), torch.rand(10, 5, 3)],
            [[torch.rand(10)]],
            torch.rand(10, 5),
            [(), []],
            [2, []],
            "",
            [[[]], []],
        ]

        if _NUMPY_AVAILABLE:
            examples += [
                np.float64(42),
                [[np.float64(42)], np.array([2])],
                np.random.rand(10, 5),
                [[np.complex128(42)], np.array([2])],
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

            # note: can_be_converted_to_tensor(example) => convertible, but not necessary equal
            assert (
                not can_be_converted_to_tensor(example) or convertible
            ), f"can_be_converted_to_tensor: {example=}"

            assert can_be_stacked(example) == stackable, f"can_be_stacked: {example=}"


class TestNDimShape(TestCase):
    def test_examples(self) -> None:
        examples = [
            [],
            (),
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
            [2, []],
            "",
            [[[]], []],
        ]
        expected_ndims = [
            1,
            1,
            2,
            2,
            3,
            3,
            2,
            3,
            3,
            3,
            2,
            2,
            ValueError,
            0,
            ValueError,
        ]
        expected_shapes = [
            (0,),
            (0,),
            (2, 3),
            ValueError,
            (2, 2, 1),
            (2, 1, 0),
            (2, 10),
            ValueError,
            (2, 10, 5),
            (1, 1, 10),
            (10, 5),
            (2, 0),
            ValueError,
            (),
            ValueError,
        ]

        if _NUMPY_AVAILABLE:
            examples += [
                np.float64(42),
                [[np.float64(42)], np.array([2])],
                np.random.rand(10, 5),
                [[np.complex128(42)], np.array([2])],
                [np.float16(42), np.float32(99)],
            ]
            expected_ndims += [0, 2, 2, 2, 1]
            expected_shapes += [(), (2, 1), (10, 5), (2, 1), (2,)]

        assert len(examples) == len(expected_ndims) == len(expected_shapes)

        for i, (example, expected_ndim, expected_shape) in enumerate(
            zip(examples, expected_ndims, expected_shapes)
        ):
            if isinstance(expected_ndim, int):
                result = ndim(example)
                assert result == expected_ndim, f"{example=}"
            else:
                with self.assertRaises(expected_ndim):
                    invalid_ndim = ndim(example)
                    print(f"{invalid_ndim=} with {example=} at {i=}")

            if isinstance(expected_shape, tuple):
                result = shape(example)
                assert result == expected_shape, f"{example=}"
            else:
                with self.assertRaises(expected_shape):
                    invalid_shape = shape(example)
                    print(f"{invalid_shape=} with {example=} at {i=}")


if __name__ == "__main__":
    unittest.main()
