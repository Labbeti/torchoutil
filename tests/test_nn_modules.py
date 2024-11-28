#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.modules import (
    Abs,
    Angle,
    AsTensor,
    CropDim,
    CropDims,
    Identity,
    LogSoftmaxMultidim,
    Mean,
    PadAndStackRec,
    PadDim,
    PadDims,
    Permute,
    RepeatInterleaveNd,
    ResampleNearestRates,
    SoftmaxMultidim,
    ToList,
    Transpose,
    Unsqueeze,
)
from torchoutil.nn.modules.mixins import ESequential


class TestSequential(TestCase):
    def test_example_1(self) -> None:
        target_length = 10
        transform = ESequential(
            ResampleNearestRates(0.5),
            PadDim(target_length),
            ResampleNearestRates(2.5),
            CropDim(target_length),
        )

        x = torch.rand(10, 20, target_length)
        result = transform(x)

        assert x.shape == result.shape

    def test_example_2(self) -> None:
        transform = ESequential(
            ToList(),
            AsTensor(),
            Abs(),
            Angle(),
            PadAndStackRec(0.0),
            PadDims([10]),
            CropDims([10]),
            Mean(dim=1),
            Unsqueeze(dim=1),
            Permute(1, 0),
            RepeatInterleaveNd(10, 0),
            Transpose(0, 1),
            LogSoftmaxMultidim(dims=(0, 1)),
            SoftmaxMultidim(dims=(1,)),
        )

        x = torch.rand(16, 10)
        result = transform(x)

        assert x.shape == result.shape

    def test_example_3(self) -> None:
        transform = ESequential(
            ToList(),
            Identity(),
            AsTensor(),
        )

        x = torch.rand(16, 10)
        result = transform(x)

        assert x.shape == result.shape


if __name__ == "__main__":
    unittest.main()
