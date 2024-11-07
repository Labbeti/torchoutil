#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

import pyoutil as po
from torchoutil.core.packaging import _NUMPY_AVAILABLE
from torchoutil.extras.numpy import np
from torchoutil.nn.functional.transform import (
    flatten,
    repeat_interleave_nd,
    resample_nearest_rates,
)


class TestRepeat(TestCase):
    def test_example_1(self) -> None:
        x = torch.as_tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        result = repeat_interleave_nd(x, repeats=2, dim=0)
        expected = torch.as_tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7]]
        )
        assert torch.equal(result, expected)


class TestResampleNearest(TestCase):
    def test_example_1(self) -> None:
        x = torch.arange(10, 20)
        result = resample_nearest_rates(x, 0.5)
        expected = torch.as_tensor([10, 12, 14, 16, 18])
        assert torch.equal(result, expected)

    def test_example_2(self) -> None:
        x = torch.arange(10, 20)
        result = resample_nearest_rates(x, 2)
        expected = torch.as_tensor(
            [
                10,
                10,
                11,
                11,
                12,
                12,
                13,
                13,
                14,
                14,
                15,
                15,
                16,
                16,
                17,
                17,
                18,
                18,
                19,
                19,
            ]
        )
        assert torch.equal(result, expected)

    def test_example_3(self) -> None:
        x = torch.stack([torch.arange(10, 20), torch.arange(20, 30)])
        result = resample_nearest_rates(x, 0.5)
        expected = torch.as_tensor([[10, 12, 14, 16, 18], [20, 22, 24, 26, 28]])
        assert torch.equal(result, expected)


class TestFlatten(TestCase):
    def test_example_1(self) -> None:
        if not _NUMPY_AVAILABLE:
            return None

        x = [
            [
                [[np.float32(64), 3.0, 0, 1], ["a", None, "", 2], torch.zeros(4)],
                torch.ones(3, 4),
            ],
        ]
        expected = [
            [np.float32(64), 3.0, 0, 1, "a", None, "", 2]
            + list(torch.zeros(4))
            + list(torch.ones(12))
        ]

        assert len(x) == len(expected)
        for xi, expected_i in zip(x, expected):
            result_i = flatten(xi)
            assert result_i == expected_i

    def test_example_2_between_dims(self) -> None:
        if not _NUMPY_AVAILABLE:
            return None

        shape = (10, 3, 4, 5)

        x = [
            (np.zeros(shape), 0, len(shape)),
            (np.zeros(shape), 1, 2),
            (np.zeros(shape), 0, 1),
            (np.float64(1.0), 0, None),
        ]
        expected = [
            np.zeros(po.prod(shape)),
            np.zeros((10, 12, 5)),
            np.zeros((30, 4, 5)),
            np.ones(1, dtype=np.float64),
        ]

        assert len(x) == len(expected)
        for (xi, start, end), expected_i in zip(x, expected):
            result_i = flatten(xi, start, end)
            assert np.equal(result_i, expected_i).all()


if __name__ == "__main__":
    unittest.main()
