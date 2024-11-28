#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

import torch

from torchoutil.types import tensor_subclasses
from torchoutil.types.tensor_subclasses import (
    BoolTensor,
    BoolTensor1D,
    ByteTensor,
    CDoubleTensor,
    CFloatTensor,
    CHalfTensor,
    ComplexFloatingTensor,
    ComplexFloatingTensor1D,
    DoubleTensor,
    FloatingTensor,
    FloatingTensor0D,
    FloatTensor,
    FloatTensor0D,
    FloatTensor1D,
    FloatTensor2D,
    HalfTensor,
    IntTensor,
    IntTensor0D,
    LongTensor,
    LongTensor0D,
    LongTensor1D,
    ShortTensor,
    ShortTensor1D,
    SignedIntegerTensor,
    SignedIntegerTensor3D,
    Tensor0D,
    Tensor1D,
    UnsignedIntegerTensor,
    _TensorNDBase,
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
        assert not isinstance(x, str)
        assert not isinstance(x, float)

    def test_subclassing_checks(self) -> None:
        assert issubclass(Tensor0D, torch.Tensor)
        assert issubclass(FloatTensor, torch.Tensor)
        assert issubclass(FloatTensor0D, torch.Tensor)

        assert issubclass(FloatTensor0D, Tensor0D)
        assert issubclass(FloatTensor0D, FloatTensor)

        assert not issubclass(FloatTensor0D, IntTensor0D)
        assert not issubclass(FloatTensor0D, FloatTensor1D)
        assert not issubclass(FloatTensor0D, str)
        assert not issubclass(FloatTensor0D, float)

        assert not issubclass(torch.Tensor, FloatTensor)
        assert not issubclass(torch.Tensor, Tensor0D)
        assert not issubclass(torch.Tensor, FloatTensor0D)
        assert not issubclass(torch.Tensor, IntTensor)

        assert issubclass(ByteTensor, UnsignedIntegerTensor)
        assert issubclass(BoolTensor, UnsignedIntegerTensor)

        assert not issubclass(ByteTensor, SignedIntegerTensor)
        assert not issubclass(ByteTensor, FloatingTensor)
        assert not issubclass(ByteTensor, ComplexFloatingTensor)

        assert not issubclass(BoolTensor, SignedIntegerTensor)
        assert not issubclass(BoolTensor, FloatingTensor)
        assert not issubclass(BoolTensor, ComplexFloatingTensor)

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

        x = ComplexFloatingTensor1D()
        assert (
            x.ndim == 1
            and not x.is_floating_point()
            and x.is_complex()
            and x.is_signed()
        )

        x = FloatingTensor0D()
        assert (
            x.ndim == 0
            and x.is_floating_point()
            and not x.is_complex()
            and x.is_signed()
        )

        x = SignedIntegerTensor3D()
        assert (
            x.ndim == 3
            and not x.is_floating_point()
            and not x.is_complex()
            and x.is_signed()
        )

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
            ShortTensor1D([1, 2], dtype=torch.uint8)

        with self.assertRaises(ValueError):
            ComplexFloatingTensor(dtype=torch.float32)

        with self.assertRaises(ValueError):
            FloatingTensor(dtype=torch.int32)

        with self.assertRaises(ValueError):
            SignedIntegerTensor(dtype=torch.complex64)

        with self.assertRaises(ValueError):
            SignedIntegerTensor(dtype=torch.bool)

        _ = ComplexFloatingTensor(dtype=torch.complex128)
        _ = FloatingTensor(dtype=torch.half)
        _ = SignedIntegerTensor(dtype=torch.long)

        x = Tensor0D(10.0)
        assert x.ndim == 0 and x.dtype == torch.float

    def test_instantiate_all(self) -> None:
        module = tensor_subclasses
        for name in dir(module):
            elem = getattr(module, name)
            if (
                not isinstance(elem, type)
                or not issubclass(elem, _TensorNDBase)
                or elem is _TensorNDBase
            ):
                continue

            tensor_cls = elem
            tensor = tensor_cls()

            assert isinstance(tensor, torch.Tensor)

    def test_complex_class(self) -> None:
        cls = ComplexFloatingTensor
        assert not cls().is_floating_point()
        assert cls().is_complex()
        assert cls().is_signed()
        assert isinstance(cls(), cls)

        assert issubclass(CFloatTensor, cls)
        assert issubclass(CHalfTensor, cls)
        assert issubclass(CDoubleTensor, cls)

        assert isinstance(torch.rand(10, dtype=torch.complex64), cls)
        assert isinstance(CFloatTensor(), cls)

        assert not isinstance(torch.rand(10), cls)
        assert not isinstance(torch.randint(0, 10, (5,)), cls)
        assert not isinstance(FloatTensor(), cls)
        assert not isinstance(DoubleTensor(), cls)

        assert not isinstance(LongTensor(), cls)

        assert not isinstance(BoolTensor1D(), cls)

    def test_floating_class(self) -> None:
        cls = FloatingTensor
        assert cls().is_floating_point()
        assert not cls().is_complex()
        assert cls().is_signed()
        assert isinstance(cls(), cls)

        assert issubclass(FloatTensor, cls)
        assert issubclass(HalfTensor, cls)
        assert issubclass(DoubleTensor, cls)

        assert not isinstance(torch.rand(10, dtype=torch.complex64), cls)
        assert not isinstance(CFloatTensor(), cls)

        assert isinstance(torch.rand(10), cls)
        assert isinstance(FloatTensor(), cls)
        assert isinstance(DoubleTensor(), cls)

        assert not isinstance(torch.randint(0, 10, (5,)), cls)
        assert not isinstance(LongTensor(), cls)

        assert not isinstance(BoolTensor1D(), cls)

    def test_integral_class(self) -> None:
        cls = SignedIntegerTensor
        assert not cls().is_floating_point()
        assert not cls().is_complex()
        assert cls().is_signed()
        assert isinstance(cls(), cls)

        assert issubclass(IntTensor, cls)
        assert issubclass(ShortTensor, cls)
        assert issubclass(LongTensor, cls)

        assert not isinstance(torch.rand(10, dtype=torch.complex64), cls)
        assert not isinstance(CFloatTensor(), cls)

        assert not isinstance(torch.rand(10), cls)
        assert not isinstance(FloatTensor(), cls)
        assert not isinstance(DoubleTensor(), cls)

        assert isinstance(torch.randint(0, 10, (5,)), cls)
        assert isinstance(LongTensor(), cls)

        assert not isinstance(BoolTensor1D(), cls)


if __name__ == "__main__":
    unittest.main()
