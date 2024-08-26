#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.types.tensor_typing import (
    FloatTensor,
    FloatTensor0D,
    IntTensor,
    IntTensor0D,
    Tensor,
    Tensor0D,
    Tensor1D,
)


class TestTensorTyping(TestCase):
    def test_example_1(self) -> None:
        x = torch.rand((), dtype=torch.float)

        assert isinstance(x, Tensor)
        assert isinstance(x, Tensor0D)
        assert isinstance(x, FloatTensor)
        assert isinstance(x, FloatTensor0D)

        assert not isinstance(x, Tensor1D)
        assert not isinstance(x, IntTensor)
        assert not isinstance(x, IntTensor0D)

    def test_example_2(self) -> None:
        assert issubclass(Tensor0D, Tensor)
        assert issubclass(FloatTensor, Tensor)
        assert issubclass(FloatTensor0D, Tensor)

        # note: however, torchoutil.FloatTensor != torch.FloatTensor and issubclass(torchoutil.FloatTensor, torch.FloatTensor) is False because toch.FloatTensor cannot be subclassed


if __name__ == "__main__":
    unittest.main()
