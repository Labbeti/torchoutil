#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

import torch

from torchoutil.types.tensor_typing import (
    DoubleTensor,
    FloatTensor,
    FloatTensor0D,
    FloatTensor1D,
    FloatTensor2D,
    IntTensor,
    IntTensor0D,
    LongTensor,
    LongTensor0D,
    LongTensor1D,
    ShortTensor1D,
    Tensor0D,
    Tensor1D,
)


class TestTensorTyping(TestCase):
    def test_instance_checks(self) -> None:
        x = torch.rand((), dtype=torch.float)

        assert isinstance(x, torch.Tensor)
        assert isinstance(x, Tensor0D)
        assert isinstance(x, FloatTensor)
        assert isinstance(x, FloatTensor0D)

        assert not isinstance(x, Tensor1D)
        assert not isinstance(x, IntTensor)
        assert not isinstance(x, IntTensor0D)

    def test_subclassing_checks(self) -> None:
        assert issubclass(Tensor0D, torch.Tensor)
        assert issubclass(FloatTensor, torch.Tensor)
        assert issubclass(FloatTensor0D, torch.Tensor)

        assert issubclass(FloatTensor0D, Tensor0D)
        assert issubclass(FloatTensor0D, FloatTensor)

        assert not issubclass(FloatTensor0D, IntTensor0D)
        assert not issubclass(FloatTensor0D, FloatTensor1D)

        assert not issubclass(torch.Tensor, FloatTensor)
        assert not issubclass(torch.Tensor, Tensor0D)
        assert not issubclass(torch.Tensor, FloatTensor0D)
        assert not issubclass(torch.Tensor, IntTensor)

    def test_default_dtype(self) -> None:
        x = [random.randint(0, 100) for _ in range(10)]

        assert Tensor1D(x).dtype == torch.long  # torch.long is the inferred dtype for x
        assert FloatTensor(x).dtype == torch.float
        assert DoubleTensor(x).dtype == torch.double
        assert LongTensor(x).dtype == torch.long

    def test_torch_floattensor_compat(self) -> None:
        args_lst = [
            (),
            (1, 2),
            (0,),
            (0, 0),
            (1, 0, 2),
            ([1, 2, 3],),
            ([],),
        ]
        for args in args_lst:
            x = DoubleTensor(*args)
            x_pt = torch.DoubleTensor(*args)

            assert isinstance(x, DoubleTensor)
            assert isinstance(x, torch.DoubleTensor)
            assert isinstance(x_pt, DoubleTensor)
            assert isinstance(x_pt, torch.DoubleTensor)

            assert (
                x.shape == x_pt.shape
                and x.dtype == x_pt.dtype
                and x.device == x_pt.device
            ), f"{args=}"

    def test_instantiation_no_args(self) -> None:
        x = Tensor0D()
        assert x.dtype == torch.get_default_dtype() and x.ndim == 0

        x = FloatTensor()
        assert x.dtype == torch.float

        x = FloatTensor0D()
        assert x.dtype == torch.float and x.ndim == 0

        x = FloatTensor1D()
        assert x.dtype == torch.float and x.ndim == 1

        x = LongTensor()
        assert x.dtype == torch.long

        x = LongTensor0D()
        assert x.dtype == torch.long and x.ndim == 0

        x = LongTensor1D()
        assert x.dtype == torch.long and x.ndim == 1

    def test_instantiation_with_args(self) -> None:
        args = (1, 2, 3)
        kwargs = dict()
        x = LongTensor(*args, **kwargs)
        assert x.ndim == len(args) and x.dtype == torch.long

        with self.assertRaises(ValueError):
            FloatTensor1D([1, 2], dtype=torch.float16)

        with self.assertRaises(ValueError):
            FloatTensor1D(1, 2)

        with self.assertRaises(ValueError):
            FloatTensor2D([2], 1)  # type: ignore

        with self.assertRaises(ValueError):
            Tensor0D([1, 2], dtype=torch.float16)

        with self.assertRaises(ValueError):
            ShortTensor1D([1, 2], dtype=torch.uint16)

        x = Tensor0D(10.0)
        assert x.ndim == 0 and x.dtype == torch.float


if __name__ == "__main__":
    unittest.main()
