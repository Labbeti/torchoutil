#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.core.packaging import _NUMPY_AVAILABLE
from torchoutil.extras.numpy import merge_numpy_dtypes, np


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
            [np.dtype("<U2"), np.dtype("<U10"), empty],
        ]
        expected_lst = [
            np.int64,
            np.complex128,
            empty,
            np.float64,
            np.dtype("<U10"),
        ]

        for args, expected in zip(args_lst, expected_lst):
            result = merge_numpy_dtypes(args, empty=empty)
            assert result == expected, f"{result=}; {expected=}"

    def test_result_dtype_compat(self) -> None:
        args_lst = [
            [np.int16, np.int32, np.int64],
            [np.complex64, np.float16, np.float64],
            [np.int64, np.float16],
            [np.dtype("<U2"), np.dtype("<U10")],
            [np.int32, np.object_],
        ]

        for args in args_lst:
            result = merge_numpy_dtypes(args)
            expected = np.result_type(*args)
            assert result == expected


if __name__ == "__main__":
    unittest.main()
