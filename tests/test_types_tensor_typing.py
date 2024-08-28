#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

import torch

from torchoutil.types.tensor_typing import (
    DoubleTensor,
    FloatTensor,
    FloatTensor0D,
    FloatTensor1D,
    IntTensor,
    IntTensor0D,
    LongTensor,
    Tensor0D,
    Tensor1D,
)


class TestTensorTyping(TestCase):
    def test_instance_checks(self) -> None:
        x = torch.rand((), dtype=torch.float)

        assert isinstance(x, torch.Tensor)
        assert isinstance(x, Tensor0D)
        assert isinstance(x, FloatTensor)
        assert isinstance(x, FloatTensor0D)

        assert not isinstance(x, Tensor1D)
        assert not isinstance(x, IntTensor)
        assert not isinstance(x, IntTensor0D)

    def test_subclassing_checks(self) -> None:
        assert issubclass(Tensor0D, torch.Tensor)
        assert issubclass(FloatTensor, torch.Tensor)
        assert issubclass(FloatTensor0D, torch.Tensor)

        assert issubclass(FloatTensor0D, Tensor0D)
        assert issubclass(FloatTensor0D, FloatTensor)

        assert not issubclass(FloatTensor0D, IntTensor0D)
        assert not issubclass(FloatTensor0D, FloatTensor1D)

        assert not issubclass(torch.Tensor, FloatTensor)
        assert not issubclass(torch.Tensor, Tensor0D)
        assert not issubclass(torch.Tensor, FloatTensor0D)
        assert not issubclass(torch.Tensor, IntTensor)

    def test_default_dtype(self) -> None:
        x = [random.randint(0, 100) for _ in range(10)]

        assert Tensor1D(x).dtype == torch.long  # torch.long is the inferred dtype for x
        assert FloatTensor(x).dtype == torch.float
        assert DoubleTensor(x).dtype == torch.double
        assert LongTensor(x).dtype == torch.long

    def test_torch_floattensor_compat(self) -> None:
        x = FloatTensor()

        assert isinstance(x, FloatTensor)
        assert isinstance(x, Tensor0D)
        assert isinstance(x, FloatTensor0D)
        assert isinstance(x, torch.FloatTensor)


if __name__ == "__main__":
    unittest.main()
