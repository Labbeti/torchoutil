#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from pathlib import Path
from unittest import TestCase

import torch

from torchoutil import Tensor, nn


class MyModule(nn.EModule[Tensor, Tensor]):
    def __init__(self, in_features: int, out_features: int, p: float) -> None:
        projection = nn.Linear(in_features, out_features)
        dropout = nn.Dropout(p=p)

        super().__init__()
        self.projection = projection
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.projection(x))


class TestInheritEModule(TestCase):
    def test_example_1(self) -> None:
        bsize = 8
        in_features = 2**5
        out_features = 2**4

        module = MyModule(in_features, out_features, p=0.5)
        x = torch.rand(bsize, in_features)
        result = module(x)

        assert tuple(result.shape) == (bsize, out_features)
        assert module.get_device() == torch.device("cpu")

        expected_config = {
            "projection.in_features": 32,
            "projection.out_features": 16,
            "dropout.p": 0.5,
            "dropout.inplace": False,
        }
        assert module.config == expected_config
        assert module.count_parameters() == in_features * out_features + out_features

        path = Path("/tmp/state_dict.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            torch.save(module.state_dict(), file)
            state_dict = torch.load(path)

        module2 = MyModule(in_features, out_features, p=0.25)
        module2.load_state_dict(state_dict)


if __name__ == "__main__":
    unittest.main()
