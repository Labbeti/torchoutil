#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.core.dtype_enum import DTypeEnum


class TestDTypeEnum(TestCase):
    def test_dtype_enum_inst(self) -> None:
        dt1 = DTypeEnum.float16
        assert dt1.dtype == torch.float16 == torch.half
        assert dt1 == "float16"
        assert dt1 == "half"
        assert dt1 == "torch.float16"
        assert dt1 == "torch.half"


if __name__ == "__main__":
    unittest.main()
