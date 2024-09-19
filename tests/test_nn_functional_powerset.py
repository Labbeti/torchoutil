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
        num_classes = multilabel.shape[-1]
        max_set_size = int(multilabel.view(-1, num_classes).sum(-1).max().int().item())

        powerset = multilabel_to_powerset(
            multilabel,
            num_classes=num_classes,
            max_set_size=max_set_size,
        )
        multilabel_back = powerset_to_multilabel(
            powerset,
            num_classes=num_classes,
            max_set_size=max_set_size,
        )
        assert torch.equal(multilabel, multilabel_back)


if __name__ == "__main__":
    unittest.main()
