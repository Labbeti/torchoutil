#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.repeat import repeat_interleave_nd


class TestRepeat(TestCase):
    def test_example_1(self) -> None:
        x = torch.as_tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        result = repeat_interleave_nd(x, repeats=2, dim=0)
        expected = torch.as_tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7]]
        )
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
