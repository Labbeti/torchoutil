#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.types import np
from torchoutil.utils.data.dtype import merge_numpy_dtypes
from torchoutil.utils.packaging import _NUMPY_AVAILABLE


class TestDType(TestCase):
    def test_merge_numpy_dtypes(self) -> None:
        if not _NUMPY_AVAILABLE:
            return None

        empty = None
        args_lst = [
            [np.int16, np.int32, np.int64],
            [np.complex64, np.float16, np.float64],
            [],
            [np.int64, np.float16],
        ]
        expected_lst = [
            np.int64,
            np.complex128,
            empty,
            np.float64,
        ]

        for args, expected in zip(args_lst, expected_lst):
            result = merge_numpy_dtypes(args, empty=empty)
            assert result == expected, f"{result=}; {expected=}"


if __name__ == "__main__":
    unittest.main()
