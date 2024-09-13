#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from typing import Sized
from unittest import TestCase

import torch

from torchoutil.core.packaging import _NUMPY_AVAILABLE
from torchoutil.extras.numpy import np
from torchoutil.nn import functional as F
from torchoutil.pyoutil.typing import is_builtin_number, is_builtin_scalar
from torchoutil.types import (
    is_number_like,
    is_numpy_number_like,
    is_scalar_like,
    is_tensor0d,
)


class TestIsNumber(TestCase):
    def test_example_1(self) -> None:
        tests = [
            (1, True),
            (1.0, True),
            (False, True),
            (1j + 2, True),
            (torch.rand(()), True),
            (torch.rand((0,)), False),
            (torch.rand(10), False),
            ([], False),
            ([[]], False),
            ((), False),
            ("test", False),
            (None, False),
            ("", False),
            (b"abc", False),
            ([1, [2]], False),
            ({}, False),
            ({"a": [1, 2], "b": [3, 4]}, False),
            (set(), False),
            ((1, 2, 3), False),
            ([1.0], False),
            ([object(), [], "abc"], False),
        ]

        if _NUMPY_AVAILABLE:
            tests += [
                (np.float64(random.random()), True),
                (np.int64(random.randint(0, 100)), True),
                (np.random.random(()), True),
                (np.random.random((0,)), False),
                (np.random.random((10,)), False),
            ]

        for x, expected in tests:
            result = is_number_like(x)
            msg = f"{x=} ({is_builtin_number(x)}, {is_tensor0d(x)}, {is_numpy_number_like(x)})"
            assert result == expected, msg

            x_is_scalar = is_scalar_like(x)
            assert isinstance(x, Sized) or x_is_scalar

            try:
                ndim = F.ndim(x)
                assert x_is_scalar == (ndim == 0), f"{type(x)=} ; {x=}"

                shape = F.shape(x)
                assert x_is_scalar == (len(shape) == 0), f"{type(x)=} ; {x=}"
                assert len(shape) == ndim, f"{type(x)=} ; {x=}"

                nelements = F.nelement(x)
                assert not x_is_scalar or (nelements == 1), f"{type(x)=} ; {x=}"

                xitem = F.item(x)
                assert is_builtin_scalar(xitem), f"{x=}"
            except (ValueError, RuntimeError, TypeError):
                assert not x_is_scalar, f"{type(x)=} ; {x=}"


if __name__ == "__main__":
    unittest.main()
