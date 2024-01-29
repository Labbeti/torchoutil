#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.pad import cat_padded_batch, pad_and_stack_rec


class TestPad(TestCase):
    def test_pad_sequence_rec_example_1(self) -> None:
        sequence = [[1, 2], [3], [], [4, 5]]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.as_tensor([[1, 2], [3, 0], [0, 0], [4, 5]])
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(output.eq(output_expected).all())

    def test_pad_sequence_rec_example_2(self) -> None:
        sequence_invalid = [[1, 2, 3], 3]
        self.assertRaises(ValueError, pad_and_stack_rec, sequence_invalid, 0)

    def test_pad_sequence_rec_example_3(self) -> None:
        sequence = [[[1], [2, 3]], [[4, 5, 6], [7], []]]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.as_tensor(
            [[[1, 0, 0], [2, 3, 0], [0, 0, 0]], [[4, 5, 6], [7, 0, 0], [0, 0, 0]]]
        )
        self.assertTrue(output.eq(output_expected).all())

    def test_pad_sequence_rec_example_4(self) -> None:
        sequence = [torch.zeros(10, 2), torch.zeros(5, 2)]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.zeros(2, 10, 2)
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(output.eq(output_expected).all())

    def test_pad_sequence_rec_example_5(self) -> None:
        sequence = [torch.zeros(3, 5), torch.zeros(3, 10)]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.zeros(2, 3, 10)
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(
            output.eq(output_expected).all(), f"{output}\n{output_expected}"
        )

    def test_pad_sequence_rec_example_6(self) -> None:
        sequence = [torch.zeros(3, 2, 5), torch.zeros(3, 6, 5)]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.zeros(2, 3, 6, 5)
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(
            output.eq(output_expected).all(), f"{output}\n{output_expected}"
        )

    def test_pad_sequence_rec_example_7(self) -> None:
        sequence = [torch.zeros(4, 2, 3, 10), torch.zeros(4, 5, 5, 10)]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.zeros(2, 4, 5, 5, 10)
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(
            output.eq(output_expected).all(), f"{output}\n{output_expected}"
        )

    def test_pad_sequence_rec_example_8(self) -> None:
        sequence = [torch.zeros(5, 10, 2), torch.zeros(2, 10, 2), torch.zeros(2, 10, 2)]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.zeros(3, 5, 10, 2)
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(
            output.eq(output_expected).all(), f"{output}\n{output_expected}"
        )

    def test_pad_sequence_rec_example_9(self) -> None:
        sequence = [[torch.ones(20, 1), torch.ones(8, 2)], [torch.ones(1, 10)]]
        output = pad_and_stack_rec(sequence, 1)
        output_expected = torch.ones(2, 2, 20, 10)
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(
            output.eq(output_expected).all(), f"{output}\n{output_expected}"
        )

    def test_pad_sequence_rec_limit_1(self) -> None:
        sequence = []
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.as_tensor([])
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(output.eq(output_expected).all())

    def test_pad_sequence_rec_limit_2(self) -> None:
        sequence = [[]]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.as_tensor([[]])
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(output.eq(output_expected).all())

    def test_pad_sequence_rec_limit_3(self) -> None:
        sequence = [[[], []], [[], []], [[], []]]
        output = pad_and_stack_rec(sequence, 0)
        output_expected = torch.as_tensor([[[], []], [[], []], [[], []]])
        self.assertEqual(output.ndim, output_expected.ndim)
        self.assertTrue(output.eq(output_expected).all())

    def test_pad_sequence_rec_limit_4(self) -> None:
        sequence = [[], [[]]]
        self.assertRaises(ValueError, pad_and_stack_rec, sequence, 0)


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
        self.assertTrue(torch.equal(expected, x12))
        self.assertTrue(torch.equal(expected_lens, x12_lens))

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
        self.assertTrue(torch.equal(expected, x12))
        self.assertTrue(torch.equal(expected_lens, x12_lens))


if __name__ == "__main__":
    unittest.main()
