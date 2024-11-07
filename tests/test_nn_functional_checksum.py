#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil import pyoutil as po
from torchoutil.nn.functional.checksum import checksum


class TestChecksum(TestCase):
    def test_checksum_tensor(self) -> None:
        x = [
            torch.arange(10),
            torch.arange(10).view(2, 5),
            list(range(10)),
        ]
        csums = [checksum(xi) for xi in x]
        assert po.all_ne(csums)


if __name__ == "__main__":
    unittest.main()
