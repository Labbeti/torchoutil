#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.modules.mixins import ESequential
from torchoutil.nn.modules.multilabel import (
    IndicesToMultihot,
    IndicesToNames,
    MultihotToIndices,
    MultihotToNames,
    NamesToIndices,
    NamesToMultihot,
)


class TestMultilabel(TestCase):
    def test_example_1(self) -> None:
        for _ in range(10):
            num_steps = int(torch.randint(1, 2, ()).item())
            num_samples = int(torch.randint(1, 2, ()).item())
            num_classes = int(torch.randint(1, 20, ()).item())
            threshold = torch.rand(())
            idx_to_name = dict(zip(range(num_classes), map(str, range(num_classes))))

            multihot = torch.rand(num_steps, num_samples, num_classes).ge(threshold)

            # dummy pipeline to convert labels multiple times
            pipeline = ESequential(
                MultihotToIndices(),
                IndicesToNames(idx_to_name),
                NamesToMultihot(idx_to_name),
                MultihotToNames(idx_to_name),
                NamesToIndices(idx_to_name),
                IndicesToMultihot(num_classes),
            )
            result = pipeline(multihot)

            assert torch.equal(multihot, result), f"{multihot.shape=}"


if __name__ == "__main__":
    unittest.main()
