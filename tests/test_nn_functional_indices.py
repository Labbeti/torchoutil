#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torch

from torch import Tensor

from torchoutil.nn.functional.indices import randperm_diff


class TestIndices(TestCase):
    def test_randperm_diff(self) -> None:
        for i in range(2, 10):
            perm = randperm_diff(i)
            arange = torch.arange(0, i, device=perm.device)

            self.assertIsInstance(perm, Tensor)
            self.assertEqual(perm.shape, arange.shape)
            self.assertFalse(perm.eq(arange).any(), f"{perm}; {arange}")


if __name__ == "__main__":
    unittest.main()
