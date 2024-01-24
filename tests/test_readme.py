#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torch

from extentorch import masked_mean


class TestReadme(TestCase):
    def test_example_1(self) -> None:
        x = torch.as_tensor([1, 2, 3, 4])
        mask = torch.as_tensor([True, True, False, False])
        result = masked_mean(x, mask)
        # result contains the mean of the values marked as True: 1.5

        result = result.item()
        self.assertEqual(result, 1.5)


if __name__ == "__main__":
    unittest.main()
