#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.multiclass import indices_to_onehot


class TestMulticlass(TestCase):
    def test_indices_to_onehot_1(self) -> None:
        indices = torch.as_tensor([[0, 2, 1], [0, 0, 2]])
        expected = torch.as_tensor(
            [[[1, 0, 0], [0, 0, 1], [0, 1, 0]], [[1, 0, 0], [1, 0, 0], [0, 0, 1]]],
            dtype=torch.bool,
        )
        result = indices_to_onehot(indices, 3)
        assert torch.equal(result, expected)

    def test_indices_to_onehot_2(self) -> None:
        indices = torch.as_tensor([[0, -1, 1], [0, 0, 2], [-1, -1, -1]])
        expected = torch.as_tensor(
            [
                [[1, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.bool,
        )
        result = indices_to_onehot(indices, 3, padding_idx=-1)
        assert torch.equal(result, expected)


if __name__ == "__main__":
    unittest.main()
