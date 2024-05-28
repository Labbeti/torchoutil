#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch
from torch import Tensor

from torchoutil.nn.functional.indices import (
    get_inverse_perm,
    get_perm_indices,
    randperm_diff,
)


class TestRandpermDiff(TestCase):
    def test_example_1(self) -> None:
        for i in range(2, 10):
            perm = randperm_diff(i)
            arange = torch.arange(0, i, device=perm.device)

            self.assertIsInstance(perm, Tensor)
            self.assertEqual(perm.shape, arange.shape)
            self.assertFalse(perm.eq(arange).any(), f"{perm}; {arange}")


class TestGetInversePerm(TestCase):
    def test_example_1(self) -> None:
        n_steps = 5
        for _ in range(n_steps):
            size = int(torch.randint(0, 100, ()).item())
            perm = torch.randperm(size)
            inv_perm = get_inverse_perm(perm)

            x1 = torch.rand(size, 10)
            x2 = x1[perm]
            x3 = x2[inv_perm]

            assert torch.equal(x1, x3)


class TestGetPermIndices(TestCase):
    def test_example_1(self) -> None:
        n_steps = 10
        for _ in range(n_steps):
            size = int(torch.randint(1, 100, ()).item())

            x1 = torch.randperm(size)
            perm = torch.randperm(size)
            x2 = x1[perm]
            perm_result = get_perm_indices(x1, x2)

            assert torch.equal(x1, x2[perm_result]), f"{x1=}, {x2=}"


if __name__ == "__main__":
    unittest.main()
