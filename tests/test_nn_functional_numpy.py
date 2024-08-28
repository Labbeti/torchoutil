#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.numpy import numpy_to_tensor, tensor_to_numpy
from torchoutil.utils.packaging import _NUMPY_AVAILABLE


class TestNumpyConversions(TestCase):
    def test_example_1(self) -> None:
        if not _NUMPY_AVAILABLE:
            return None

        x_tensor = torch.rand(3, 4, 5)
        x_array = tensor_to_numpy(x_tensor)
        result = numpy_to_tensor(x_array)

        assert torch.equal(x_tensor, result)


if __name__ == "__main__":
    unittest.main()
