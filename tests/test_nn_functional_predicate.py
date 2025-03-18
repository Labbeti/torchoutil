#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import warnings
from unittest import TestCase

import torch

from torchoutil.core.packaging import _NUMPY_AVAILABLE
from torchoutil.extras.numpy import np
from torchoutil.nn.functional.predicate import (
    all_eq,
    is_convertible_to_tensor,
    is_stackable,
)


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
            warnings.filterwarnings("ignore", category=UserWarning)
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

            # note: is_convertible_to_tensor(example) => convertible, but not necessary equal
            assert (
                not is_convertible_to_tensor(example) or convertible
            ), f"can_be_converted_to_tensor: {example=}"

            assert is_stackable(example) == stackable, f"is_stackable: {example=}"


class TestAllEq(TestCase):
    def test_all_eq_example(self) -> None:
        x = torch.full((100, 2, 4), torch.rand(()).item())
        assert all_eq(x)

    def test_all_eq_dim(self) -> None:
        x = torch.as_tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ]
        )

        assert not all_eq(x)
        assert all_eq(x, dim=0).all()
        assert not all_eq(x, dim=1).any()

        assert not all_eq(x.T)
        assert not all_eq(x.T, dim=0).any()
        assert all_eq(x.T, dim=1).all()

        if _NUMPY_AVAILABLE:
            x = x.numpy()
            assert not all_eq(x)
            assert all_eq(x, dim=0).all()
            assert not all_eq(x, dim=1).any()

            assert not all_eq(x.T)
            assert not all_eq(x.T, dim=0).any()
            assert all_eq(x.T, dim=1).all()


if __name__ == "__main__":
    unittest.main()
