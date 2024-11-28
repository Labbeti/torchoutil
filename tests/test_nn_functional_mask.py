#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.mask import (
    generate_square_subsequent_mask,
    lengths_to_non_pad_mask,
    lengths_to_pad_mask,
    non_pad_mask_to_lengths,
    pad_mask_to_lengths,
    tensor_to_non_pad_mask,
    tensor_to_pad_mask,
)


class TestTensorToPadMask(TestCase):
    def test_tensor_to_non_pad_mask_example_1(self) -> None:
        inp = torch.as_tensor([1, 10, 20, 2, 0, 0])
        # default include_end value is False
        out = tensor_to_non_pad_mask(inp, end_value=2)
        out_expected = torch.as_tensor([True, True, True, False, False, False])

        assert out.shape == out_expected.shape
        assert torch.equal(out, out_expected)

    def test_tensor_to_non_pad_mask_example_2(self) -> None:
        inp = torch.as_tensor([1, 10, 20, 2, 0, 0])
        out = tensor_to_non_pad_mask(inp, end_value=2, include_end=True)
        out_expected = torch.as_tensor([True, True, True, True, False, False])

        assert out.shape == out_expected.shape
        assert torch.equal(out, out_expected)

    def test_tensor_to_non_pad_mask_example_3(self) -> None:
        inp = torch.as_tensor(
            [
                [10, 20, 2, 0, 2],
                [10, 20, 30, 0, 0],
                [2, 2, 0, 99, 10],
            ]
        )
        out = tensor_to_non_pad_mask(inp, end_value=2, include_end=False)
        out_expected = torch.as_tensor(
            [
                [True, True, False, False, False],
                [True, True, True, True, True],
                [False, False, False, False, False],
            ]
        )

        assert out.shape == out_expected.shape
        assert torch.equal(out, out_expected)

    def test_tensor_to_pad_mask_example_1(self) -> None:
        inp = torch.as_tensor([1, 10, 20, 2, 0, 0])
        out_expected = torch.as_tensor([False, False, False, True, True, True])
        out = tensor_to_pad_mask(inp, end_value=2)  # default include_end value is True

        assert out.shape == out_expected.shape
        assert torch.equal(out, out_expected)

    def test_tensor_to_pad_mask_example_2(self) -> None:
        inp = torch.as_tensor([1, 10, 20, 2, 0, 0])
        out_expected = torch.as_tensor([False, False, False, False, True, True])
        out = tensor_to_pad_mask(inp, end_value=2, include_end=False)

        assert out.shape == out_expected.shape
        assert torch.equal(out, out_expected)

    def test_tensor_to_pad_mask_example_3(self) -> None:
        inp = torch.as_tensor([[10, 20, 30, 40, 50], [10, 2, 20, 30, 40]])
        out_expected = torch.as_tensor(
            [[False, False, False, False, False], [False, False, True, True, True]]
        )
        out = tensor_to_pad_mask(inp, end_value=2, include_end=False)

        assert out.shape == out_expected.shape
        assert torch.equal(out, out_expected)

    def test_tensor_to_pad_mask_example_4(self) -> None:
        inp = torch.as_tensor([[10, 20, 30, 40, 50], [10, 2, 20, 30, 40]])
        out_expected = torch.as_tensor(
            [[False, False, False, False, False], [False, True, True, True, True]]
        )
        out = tensor_to_pad_mask(inp, end_value=2, include_end=True)

        assert out.shape == out_expected.shape
        assert torch.equal(out, out_expected)


class TestMaskLengths(TestCase):
    def test_mask_to_lens_to_mask(self) -> None:
        non_pad_mask = torch.as_tensor(
            [[1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.bool
        )

        out = non_pad_mask_to_lengths(non_pad_mask)
        out_expected = torch.as_tensor([2, 0, 3])
        assert torch.equal(out, out_expected)

        inp2 = out
        out2 = lengths_to_non_pad_mask(inp2, None)
        out2_expected = non_pad_mask
        assert torch.equal(out2, out2_expected), f"{out2=}\n{out2_expected=}"

        pad_mask = torch.as_tensor(
            [
                [0, 1, 1],
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 1],
            ],
            dtype=torch.bool,
        )
        lens = pad_mask_to_lengths(pad_mask)
        out3 = lengths_to_pad_mask(lens, None)
        out3_expected = pad_mask

        assert torch.equal(out3, out3_expected), f"{out3=}; {out3_expected=}"

    def test_lens_to_non_pad_masks(self) -> None:
        lens = torch.arange(5)

        # Default behaviour
        include_end = False
        expected_mask = torch.as_tensor(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.int,
        )

        non_pad_mask = lengths_to_non_pad_mask(lens, 4).int()
        assert non_pad_mask.shape == expected_mask.shape
        self.assertTrue(
            torch.equal(non_pad_mask, expected_mask),
            f"{include_end=}; {non_pad_mask=}; {expected_mask=}",
        )

        # Special behaviour
        include_end = True
        expected_mask = torch.as_tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.int,
        )

        non_pad_mask = lengths_to_non_pad_mask(lens, 4, include_end).int()
        assert non_pad_mask.shape == expected_mask.shape
        self.assertTrue(
            torch.equal(non_pad_mask, expected_mask),
            f"{include_end=}; {non_pad_mask=}; {expected_mask=}",
        )

    def test_lengths_to_non_pad_mask_example_1(self) -> None:
        input = torch.as_tensor([4, 2, 0, 3, 0])
        output = lengths_to_non_pad_mask(input, max_len=6, include_end=False)
        expected = torch.as_tensor(
            [
                [True, True, True, True, False, False],
                [True, True, False, False, False, False],
                [False, False, False, False, False, False],
                [True, True, True, False, False, False],
                [False, False, False, False, False, False],
            ]
        )

        assert output.shape == expected.shape
        assert torch.equal(output, expected)

    def test_lengths_to_pad_mask_example_1(self) -> None:
        input = torch.as_tensor([4, 2, 0, 3, 0])
        output = lengths_to_pad_mask(input, max_len=None, include_end=True)
        expected = torch.as_tensor(
            [
                [False, False, False, False],
                [False, False, True, True],
                [True, True, True, True],
                [False, False, False, True],
                [True, True, True, True],
            ]
        )

        assert output.shape == expected.shape
        assert torch.equal(output, expected)

    def test_lens_to_pad_masks(self) -> None:
        lens = torch.arange(5)

        # Default behaviour
        include_end = True
        expected_mask = torch.as_tensor(
            [
                [1, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=torch.int,
        )

        pad_mask = lengths_to_pad_mask(lens, 4).int()
        assert pad_mask.shape == expected_mask.shape
        self.assertTrue(
            torch.equal(pad_mask, expected_mask),
            f"{include_end=}; {pad_mask=}; {expected_mask=}",
        )

        # Special behaviour
        include_end = False
        expected_mask = torch.as_tensor(
            [
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.int,
        )

        pad_mask = lengths_to_pad_mask(lens, 4, include_end).int()
        assert pad_mask.shape == expected_mask.shape
        self.assertTrue(
            torch.equal(pad_mask, expected_mask),
            f"{include_end=}; {pad_mask=}; {expected_mask=}",
        )


class TestGenerateSqMask(TestCase):
    def test_generate_square_subsequent_mask_example_1(self) -> None:
        inf = math.inf
        output = generate_square_subsequent_mask(6)
        expected = torch.as_tensor(
            [
                [0.0, -inf, -inf, -inf, -inf, -inf],
                [0.0, 0.0, -inf, -inf, -inf, -inf],
                [0.0, 0.0, 0.0, -inf, -inf, -inf],
                [0.0, 0.0, 0.0, 0.0, -inf, -inf],
                [0.0, 0.0, 0.0, 0.0, 0.0, -inf],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert output.shape == expected.shape
        assert torch.equal(output, expected)

    def test_generate_shifted_sq_mask_example_1(self) -> None:
        inf = math.inf
        output = generate_square_subsequent_mask(6, diagonal=2)
        expected = torch.as_tensor(
            [
                [0.0, 0.0, 0.0, -inf, -inf, -inf],
                [0.0, 0.0, 0.0, 0.0, -inf, -inf],
                [0.0, 0.0, 0.0, 0.0, 0.0, -inf],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert output.shape == expected.shape
        assert torch.equal(output, expected)

    def test_generate_shifted_sq_mask_example_2(self) -> None:
        inf = math.inf
        output = generate_square_subsequent_mask(6, diagonal=-2)
        expected = torch.as_tensor(
            [
                [-inf, -inf, -inf, -inf, -inf, -inf],
                [-inf, -inf, -inf, -inf, -inf, -inf],
                [0.0, -inf, -inf, -inf, -inf, -inf],
                [0.0, 0.0, -inf, -inf, -inf, -inf],
                [0.0, 0.0, 0.0, -inf, -inf, -inf],
                [0.0, 0.0, 0.0, 0.0, -inf, -inf],
            ]
        )
        assert output.shape == expected.shape
        assert torch.equal(output, expected)

    def test_generate_shifted_sq_mask_example_3(self) -> None:
        inf = math.inf
        output = generate_square_subsequent_mask(6, diagonal=4)
        expected = torch.as_tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, -inf],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert output.shape == expected.shape
        assert torch.equal(output, expected)


if __name__ == "__main__":
    unittest.main()
