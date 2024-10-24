#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.core.packaging import _NUMPY_AVAILABLE
from torchoutil.extras.numpy import (
    np,
    numpy_is_complex,
    numpy_is_complex_dtype,
    numpy_to_tensor,
    numpy_view_as_complex,
    numpy_view_as_real,
    tensor_to_numpy,
)


class TestNumpyConversions(TestCase):
    def test_example_1(self) -> None:
        if not _NUMPY_AVAILABLE:
            return None

        x_tensor = torch.rand(3, 4, 5)
        x_array = tensor_to_numpy(x_tensor)
        result = numpy_to_tensor(x_array)

        assert torch.equal(x_tensor, result)

    def test_complex(self) -> None:
        if not _NUMPY_AVAILABLE:
            return None

        complex_dtypes = [np.complex64, np.complex128]
        x_complex = [
            np.array(
                np.random.rand(1) * 1j,
                dtype=complex_dtypes[np.random.randint(0, len(complex_dtypes))],
            )
            for _ in range(1000)
        ]
        assert all(numpy_is_complex(xi) for xi in x_complex)
        assert all(numpy_is_complex_dtype(xi.dtype) for xi in x_complex)

        x_real = [numpy_view_as_real(xi) for xi in x_complex]
        assert all(not numpy_is_complex(xi) for xi in x_real)
        assert all(not numpy_is_complex_dtype(xi.dtype) for xi in x_real)

        result = [numpy_view_as_complex(xi) for xi in x_real]
        assert all(numpy_is_complex(xi) for xi in result)
        assert all(numpy_is_complex_dtype(xi.dtype) for xi in result)
        assert x_complex == result


if __name__ == "__main__":
    unittest.main()
