#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torch

from torchoutil import masked_mean, insert_at_indices


class TestReadme(TestCase):
    def test_example_1(self) -> None:
        x = torch.as_tensor([1, 2, 3, 4])
        mask = torch.as_tensor([True, True, False, False])
        result = masked_mean(x, mask)
        # result contains the mean of the values marked as True: 1.5

        result = result.item()
        self.assertEqual(result, 1.5)

    def test_example_2(self) -> None:
        x = torch.as_tensor([1, 2, 3, 4])
        result = insert_at_indices(x, [0, 2], 5)
        expected = torch.as_tensor([5, 1, 2, 5, 3, 4])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
