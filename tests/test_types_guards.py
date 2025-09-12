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
from torchoutil.pyoutil.typing import (
    is_builtin_number,
    is_builtin_scalar,
    isinstance_guard,
)
from torchoutil.types import (
    Tensor0D,
    is_number_like,
    is_numpy_number_like,
    is_numpy_scalar_like,
    is_scalar_like,
)


class TestGuards(TestCase):
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
            (bytearray(), False),
            ([1, [2]], False),
            ({}, False),
            ({"a": [1, 2], "b": [3, 4]}, False),
            (set(), False),
            ((1, 2, 3), False),
            ([1.0], False),
            ([object(), [], "abc"], False),
        ]
        tests = [tuple_ + (False, False) for tuple_ in tests]

        if _NUMPY_AVAILABLE:
            tests += [
                (np.float64(random.random()), True, True, True),
                (np.int64(random.randint(0, 100)), True, True, True),
                (np.random.random(()), True, True, True),
                (np.random.random((0,)), False, False, False),
                (np.random.random((10,)), False, False, False),
                (np.array(["abc"])[0], False, False, True),
                (np.array(["abc"]), False, False, False),
            ]

        for x, expected_is_num, expected_is_np_num, expected_is_np_sca in tests:
            x_is_number = is_number_like(x)
            x_is_scalar = is_scalar_like(x)
            x_is_np_number = is_numpy_number_like(x)
            x_is_np_scalar = is_numpy_scalar_like(x)

            msg = f"{x=} ({type(x)=}, {is_builtin_number(x)=}, {isinstance_guard(x, Tensor0D)=}, {x_is_np_number=})"
            assert x_is_number == expected_is_num, msg
            assert x_is_np_number == expected_is_np_num, msg
            assert x_is_np_scalar == expected_is_np_sca, msg

            assert isinstance(x, Sized) or x_is_scalar

            # Impl: number => scalar
            assert not x_is_number or x_is_scalar
            assert not x_is_np_number or x_is_number
            assert not x_is_np_scalar or x_is_scalar

            if _NUMPY_AVAILABLE:
                np_x_is_scalar = np.isscalar(x)
                # Impl: np_scalar => scalar
                assert not np_x_is_scalar or x_is_scalar, f"{x=}"

            try:
                # Impl: scalar => (ndim == 0)
                ndim = F.get_ndim(x)  # type: ignore
                assert not x_is_scalar or (ndim == 0), f"{type(x)=} ; {x=}"

                # Impl: scalar => (len(shape) == 0)
                shape = F.get_shape(x)  # type: ignore
                assert not x_is_scalar or (len(shape) == 0), f"{type(x)=} ; {x=}"

                assert len(shape) == ndim, f"{type(x)=} ; {x=}"

                if _NUMPY_AVAILABLE:
                    np_x = np.array(x)
                    assert shape == np_x.shape, f"{x=}"

                # Impl: scalar => (nelements == 1)
                nelements = F.nelement(x)  # type: ignore
                assert not x_is_scalar or (
                    nelements == 1
                ), f"{type(x)=} ; {x=}; {x_is_scalar=}, {nelements=}"

                xitem = F.to_item(x)  # type: ignore
                assert is_builtin_scalar(xitem), f"{x=}"

            except (ValueError, RuntimeError, TypeError) as err:
                assert not x_is_scalar, f"{type(x)=} ; {x=} ; {err=}"

    def test_is_builtin_strict(self) -> None:
        class subint(int):
            ...

        i1 = 100
        i2 = subint(100)

        assert is_builtin_scalar(i1, strict=False)
        assert is_builtin_scalar(i1, strict=True)
        assert is_builtin_scalar(i2, strict=False)
        assert not is_builtin_scalar(i2, strict=True)

        if not _NUMPY_AVAILABLE:
            return None

        s1 = "abc"
        arr = np.array([s1, s1])
        s2 = arr[0]

        assert is_builtin_scalar(s1, strict=False)
        assert is_builtin_scalar(s1, strict=True)
        assert is_builtin_scalar(s2, strict=False)
        assert not is_builtin_scalar(s2, strict=True)


if __name__ == "__main__":
    unittest.main()
