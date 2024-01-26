#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torch

from torchoutil import (
    masked_mean,
    lengths_to_non_pad_mask,
    probs_to_names,
    multihot_to_indices,
    insert_at_indices,
)


class TestReadme(TestCase):
    def test_masked_mean_example(self) -> None:
        x = torch.as_tensor([1, 2, 3, 4])
        mask = torch.as_tensor([True, True, False, False])
        result = masked_mean(x, mask)
        # result contains the mean of the values marked as True: 1.5

        result = result.item()
        self.assertEqual(result, 1.5)

    def test_lengths_to_non_pad_mask_example(self) -> None:
        x = torch.as_tensor([3, 1, 2])
        pad_mask = lengths_to_non_pad_mask(x, max_len=4)
        expected = torch.as_tensor(
            [
                [True, True, True, False],
                [True, False, False, False],
                [True, True, False, False],
            ]
        )

        self.assertTrue(torch.equal(pad_mask, expected))

    def test_probs_to_names_example(self) -> None:
        probs = torch.as_tensor([[0.9, 0.1], [0.6, 0.9]])
        names = probs_to_names(probs, threshold=0.5, idx_to_name={0: "Cat", 1: "Dog"})
        expected = [["Cat"], ["Cat", "Dog"]]

        self.assertListEqual(names, expected)

    def test_multihot_to_indices_example(self) -> None:
        multihot = torch.as_tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
        indices = multihot_to_indices(multihot)
        expected = [[0], [1, 2], []]

        self.assertListEqual(indices, expected)

    def test_insert_at_indices_example(self) -> None:
        x = torch.as_tensor([1, 2, 3, 4])
        result = insert_at_indices(x, [0, 2], 5)
        expected = torch.as_tensor([5, 1, 2, 5, 3, 4])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
