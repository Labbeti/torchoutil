#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.crop import crop_dims


class TestReadme(TestCase):
    def test_crop_dims_example_1(self) -> None:
        x = torch.zeros(10, 10, 10)
        result = crop_dims(x, [1, 2, 3], dims="auto")
        expected = torch.zeros(1, 2, 3)
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
