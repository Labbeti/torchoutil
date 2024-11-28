#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
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
            [torch.arange(10)],
            list(range(10)),
            tuple(range(10)),
            set(range(10)),
            range(10),
            [],
            None,
            0,
            1,
            math.nan,
            [1, 2],
            [2, 1],
            [1, 2, 0],
            (1, 2),
        ]
        csums = [checksum(xi) for xi in x]
        assert po.all_ne(csums), f"{csums=}"


if __name__ == "__main__":
    unittest.main()
