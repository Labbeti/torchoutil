#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.modules import CropDim, PadDim, ResampleNearestRates
from torchoutil.nn.modules.mixins import TSequential


class TestTransforms(TestCase):
    def test_example_1(self) -> None:
        target_length = 10
        transform = TSequential(
            ResampleNearestRates(0.5),
            PadDim(target_length),
            ResampleNearestRates(2.5),
            CropDim(target_length),
        )

        x = torch.rand(10, 20, target_length)
        result = transform(x)

        assert x.shape == result.shape


if __name__ == "__main__":
    unittest.main()
