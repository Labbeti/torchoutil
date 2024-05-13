#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil import (
    get_inverse_perm,
    insert_at_indices,
    lengths_to_non_pad_mask,
    masked_mean,
    multihot_to_indices,
    probs_to_name,
    probs_to_names,
)


class TestReadme(TestCase):
    def test_masked_mean_example(self) -> None:
        x = torch.as_tensor([1, 2, 3, 4])
        mask = torch.as_tensor([True, True, False, False])
        result = masked_mean(x, mask)
        # result contains the mean of the values marked as True: 1.5
        result = result.item()

        assert result == 1.5

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

        assert torch.equal(pad_mask, expected)

    def test_probs_to_name_example(self) -> None:
        probs = torch.as_tensor([[0.9, 0.1], [0.4, 0.6]])
        names = probs_to_name(probs, idx_to_name={0: "Cat", 1: "Dog"})
        expected = ["Cat", "Dog"]

        assert names == expected

    def test_probs_to_names_example(self) -> None:
        probs = torch.as_tensor([[0.9, 0.1], [0.6, 0.9]])
        names = probs_to_names(probs, threshold=0.5, idx_to_name={0: "Cat", 1: "Dog"})
        expected = [["Cat"], ["Cat", "Dog"]]

        assert names == expected

    def test_multihot_to_indices_example(self) -> None:
        multihot = torch.as_tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
        indices = multihot_to_indices(multihot)
        expected = [[0], [1, 2], []]

        assert indices == expected

    def test_insert_at_indices_example(self) -> None:
        x = torch.as_tensor([1, 2, 3, 4])
        result = insert_at_indices(x, [0, 2], 5)
        expected = torch.as_tensor([5, 1, 2, 5, 3, 4])

        assert torch.equal(result, expected)

    def test_get_inverse_perm_example(self) -> None:
        perm = torch.randperm(10)
        inv_perm = get_inverse_perm(perm)

        x1 = torch.rand(10)
        x2 = x1[perm]
        x3 = x2[inv_perm]
        # inv_perm are indices that allow us to get x1 from x3, i.e. x1 == x3 here

        assert torch.equal(x1, x3)


if __name__ == "__main__":
    unittest.main()
