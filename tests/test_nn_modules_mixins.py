#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

import torch

from torchoutil import Tensor, nn
from torchoutil.nn.modules.mixins import _DEFAULT_DEVICE_DETECT_MODE


class Intermediate(torch.nn.Module):
    """Subclass of torch Module instead of torchoutil EModule here."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.intermediate_attr = "abcd"

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class MyModule(nn.EModule[Tensor, Tensor]):
    def __init__(self, in_features: int, out_features: int, p: float) -> None:
        projection = Intermediate(in_features, out_features)
        dropout = nn.Dropout(p=p)

        super().__init__()
        self.projection = projection
        self.dropout = dropout
        self.p = p
        self.sizes = (in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.projection(x))


class TestInheritEModule(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmpdir = Path(os.getenv("TORCHOUTIL_TMPDIR", tempfile.gettempdir())).joinpath(
            "torchoutil_tests"
        )
        cls.tmpdir = tmpdir

    def test_multiple(self) -> None:
        bsize = 8
        in_features = 2**5
        out_features = 2**4

        module1 = MyModule(in_features, out_features, p=0.5)
        x = torch.rand(bsize, in_features)
        result = module1(x)

        assert tuple(result.shape) == (bsize, out_features)
        assert module1.get_device() == torch.device("cpu")

        expected_config = {
            "projection.linear.in_features": 32,
            "projection.linear.out_features": 16,
            "projection.intermediate_attr": "abcd",
            "dropout.p": 0.5,
            "dropout.inplace": False,
            "p": 0.5,
            "sizes": (32, 16),
        }
        assert module1.config == expected_config
        assert module1.count_parameters() == in_features * out_features + out_features

        path = self.tmpdir.joinpath("state_dict.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            torch.save(module1.state_dict(), file)
            state_dict = torch.load(path)

        module2 = MyModule(in_features, out_features, p=0.25)
        module2.load_state_dict(state_dict)

        assert module1.checksum() == module2.checksum()

    def test_example_device_detect_mode_default(self) -> None:
        module = MyModule(10, 16, p=0.0)
        assert module.device_detect_mode == _DEFAULT_DEVICE_DETECT_MODE


if __name__ == "__main__":
    unittest.main()
