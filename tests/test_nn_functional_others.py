#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torch

from extentorch.nn.functional.others import cat_padded_batch


class TestCatPaddedBatch(TestCase):
    def test_example_1(self) -> None:
        # Cas 2D
        x1 = torch.as_tensor(
            [
                [1, 1, 0, 0],
                [2, 2, 2, 0],
                [3, 0, 0, 0],
            ]
        )
        x2 = torch.as_tensor(
            [
                [4, 4, 4, 4, 4],
                [5, 5, 0, 0, 0],
                [6, 6, 0, 0, 0],
            ]
        )
        x1_lens = torch.as_tensor([2, 3, 1])
        x2_lens = torch.as_tensor([5, 2, 2])
        seq_dim = -1
        batch_dim = 0

        expected = torch.as_tensor(
            [
                [1, 1, 4, 4, 4, 4, 4],
                [2, 2, 2, 5, 5, 0, 0],
                [3, 6, 6, 0, 0, 0, 0],
            ]
        )
        expected_lens = torch.as_tensor([7, 5, 3])
        x12, x12_lens = cat_padded_batch(x1, x1_lens, x2, x2_lens, seq_dim, batch_dim)

        self.assertEqual(expected.shape, x12.shape)
        self.assertEqual(expected_lens.shape, x12_lens.shape)
        self.assertTrue(expected.eq(x12).all().item())
        self.assertTrue(expected_lens.eq(x12_lens).all().item())

    def test_example_2(self) -> None:
        # Cas 3D
        x1 = torch.as_tensor(
            [
                [[1, 11], [1, 11], [0, 00], [0, 00]],
                [[2, 22], [2, 22], [2, 22], [0, 00]],
                [[3, 33], [0, 00], [0, 00], [0, 00]],
            ]
        )
        x2 = torch.as_tensor(
            [
                [[4, 44], [4, 44], [4, 44], [4, 44], [4, 44]],
                [[5, 55], [5, 55], [0, 00], [0, 00], [0, 00]],
                [[6, 66], [6, 66], [0, 00], [0, 00], [0, 00]],
            ]
        )
        x1_lens = torch.as_tensor([2, 3, 1])
        x2_lens = torch.as_tensor([5, 2, 2])
        seq_dim = -2
        batch_dim = 0

        expected = torch.as_tensor(
            [
                [[1, 11], [1, 11], [4, 44], [4, 44], [4, 44], [4, 44], [4, 44]],
                [[2, 22], [2, 22], [2, 22], [5, 55], [5, 55], [0, 00], [0, 00]],
                [[3, 33], [6, 66], [6, 66], [0, 00], [0, 00], [0, 00], [0, 00]],
            ]
        )
        expected_lens = torch.as_tensor([7, 5, 3])
        x12, x12_lens = cat_padded_batch(x1, x1_lens, x2, x2_lens, seq_dim, batch_dim)

        self.assertEqual(expected.shape, x12.shape)
        self.assertEqual(expected_lens.shape, x12_lens.shape)
        self.assertTrue(expected.eq(x12).all().item())
        self.assertTrue(expected_lens.eq(x12_lens).all().item())


if __name__ == "__main__":
    unittest.main()
