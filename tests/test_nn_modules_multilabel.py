#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch
from torch import Tensor

from torchoutil.nn.modules.multilabel import (
    IndicesToMultihot,
    IndicesToNames,
    MultihotToIndices,
    MultihotToNames,
    NamesToIndices,
    NamesToMultihot,
)
from torchoutil.nn.modules.typed import TSequential


class TestMultilabel(TestCase):
    def test_modules(self) -> None:
        num_samples = int(torch.randint(0, 20, ()).item())
        num_classes = int(torch.randint(1, 20, ()).item())
        threshold = torch.rand(())
        idx_to_name = dict(zip(range(num_classes), map(str, range(num_classes))))

        multihot = torch.rand(num_samples, num_classes).ge(threshold)

        # dummy pipeline to convert labels multiple times
        pipeline = TSequential[Tensor, Tensor](
            MultihotToIndices(),
            IndicesToNames(idx_to_name),
            NamesToMultihot(idx_to_name),
            MultihotToNames(idx_to_name),
            NamesToIndices(idx_to_name),
            IndicesToMultihot(num_classes),
        )
        result = pipeline(multihot)
        self.assertTrue(torch.equal(multihot, result))


if __name__ == "__main__":
    unittest.main()
