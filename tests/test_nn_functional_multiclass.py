#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.multiclass import index_to_onehot, onehot_to_index


class TestMulticlass(TestCase):
    def test_index_to_onehot_1(self) -> None:
        indices = torch.as_tensor([[0, 2, 1], [0, 0, 2]])
        expected = torch.as_tensor(
            [[[1, 0, 0], [0, 0, 1], [0, 1, 0]], [[1, 0, 0], [1, 0, 0], [0, 0, 1]]],
            dtype=torch.bool,
        )
        result = index_to_onehot(indices, 3)
        assert torch.equal(result, expected)

    def test_index_to_onehot_2(self) -> None:
        indices = torch.as_tensor([[0, -1, 1], [0, 0, 2], [-1, -1, -1]])
        expected = torch.as_tensor(
            [
                [[1, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.bool,
        )
        result = index_to_onehot(indices, 3, padding_idx=-1)
        assert torch.equal(result, expected)

    def test_onehot_to_index_3d(self) -> None:
        onehot = torch.as_tensor([[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]]])
        expected = [[1, 0], [2, 2]]
        result = onehot_to_index(onehot)
        assert result == expected

    def test_index_to_onehot_3d(self) -> None:
        indices = [[1, 0], [2, 2]]
        expected = torch.as_tensor(
            [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1]]]
        ).bool()
        result = index_to_onehot(indices, 3)
        assert torch.equal(result, expected)


if __name__ == "__main__":
    unittest.main()
