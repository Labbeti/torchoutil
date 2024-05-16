#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.transform import (
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


if __name__ == "__main__":
    unittest.main()
