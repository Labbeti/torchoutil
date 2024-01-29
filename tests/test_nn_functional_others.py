#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.others import can_be_converted_to_tensor


class TestOthers(TestCase):
    def test_can_be_converted_to_tensor_1(self) -> None:
        lst = [[1, 0, 0], [2, 3, 4]]
        self.assertTrue(can_be_converted_to_tensor(lst))

    def test_can_be_converted_to_tensor_2(self) -> None:
        lst = [[1, 0, 0], [2, 3]]
        self.assertFalse(can_be_converted_to_tensor(lst))

    def test_can_be_converted_to_tensor_3(self) -> None:
        lst = [[[True], [False]], [[False], [True]]]
        self.assertTrue(can_be_converted_to_tensor(lst))

    def test_can_be_converted_to_tensor_4(self) -> None:
        lst = [[[]], [[]]]
        self.assertTrue(can_be_converted_to_tensor(lst))

    def test_can_be_converted_to_tensor_5(self) -> None:
        lst = [torch.rand(10), torch.rand(10)]
        self.assertFalse(can_be_converted_to_tensor(lst))


if __name__ == "__main__":
    unittest.main()
