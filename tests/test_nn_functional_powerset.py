#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.powerset import (
    multilabel_to_powerset,
    powerset_to_multilabel,
)


class TestPowerset(TestCase):
    def test_example_1(self) -> None:
        multilabel = torch.as_tensor(
            [[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 0]]
        )[None].float()
        powerset = multilabel_to_powerset(multilabel, num_classes=4, max_set_size=2)
        multilabel_back = powerset_to_multilabel(
            powerset, num_classes=4, max_set_size=2
        )
        assert torch.equal(multilabel, multilabel_back)


if __name__ == "__main__":
    unittest.main()
