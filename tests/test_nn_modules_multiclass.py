#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.modules.mixins import ESequential
from torchoutil.nn.modules.multiclass import (
    IndexToName,
    IndexToOnehot,
    NameToIndex,
    NameToOnehot,
    OnehotToIndex,
    OnehotToName,
    ProbsToOnehot,
)


class TestMulticlass(TestCase):
    def test_example_1(self) -> None:
        for _ in range(10):
            num_steps = int(torch.randint(0, 10, ()).item())
            num_samples = int(torch.randint(0, 20, ()).item())
            num_classes = int(torch.randint(1, 20, ()).item())
            idx_to_name = dict(zip(range(num_classes), map(str, range(num_classes))))

            probs = torch.rand(num_steps, num_samples, num_classes)
            onehot = ProbsToOnehot()(probs)

            # dummy pipeline to convert labels multiple times
            pipeline = ESequential(
                OnehotToName(idx_to_name),
                NameToIndex(idx_to_name),
                IndexToOnehot(len(idx_to_name)),
                OnehotToIndex(),
                IndexToName(idx_to_name),
                NameToOnehot(idx_to_name),
            )
            result = pipeline(onehot)

            assert torch.equal(onehot, result)


if __name__ == "__main__":
    unittest.main()
