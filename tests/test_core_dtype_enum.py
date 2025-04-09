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

    def test_check_properties(self) -> None:
        dt = torch.complex128
        dte = DTypeEnum.from_dtype(dt)

        assert dt.is_complex == dte.is_complex
        assert dt.is_floating_point == dte.is_floating_point
        assert dt.is_signed == dte.is_signed

        try:
            expected = dt.itemsize  # type: ignore
        except AttributeError as err:
            expected = err

        try:
            result = dte.itemsize
        except AttributeError as err:
            result = err

        assert (
            isinstance(expected, AttributeError) and isinstance(result, AttributeError)
        ) or (expected == result)

        try:
            expected = dt.to_real()  # type: ignore
        except AttributeError as err:
            expected = err

        try:
            result = dte.to_real()
        except AttributeError as err:
            result = err

        assert (
            isinstance(expected, AttributeError) and isinstance(result, AttributeError)
        ) or (expected == result)


if __name__ == "__main__":
    unittest.main()
