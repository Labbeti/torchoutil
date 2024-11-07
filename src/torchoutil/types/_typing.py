#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TypeVar, Union

import torch

from torchoutil.extras.numpy.definitions import (  # noqa: F401
    NumpyNumberLike,
    NumpyScalarLike,
    np,
    numpy,
)
from torchoutil.pyoutil.typing.classes import BuiltinNumber, BuiltinScalar

from .tensor_subclasses import (  # noqa: F401
    BoolTensor,
    BoolTensor0D,
    BoolTensor1D,
    BoolTensor2D,
    BoolTensor3D,
    ByteTensor,
    ByteTensor0D,
    ByteTensor1D,
    ByteTensor2D,
    ByteTensor3D,
    CDoubleTensor,
    CDoubleTensor0D,
    CDoubleTensor1D,
    CDoubleTensor2D,
    CDoubleTensor3D,
    CFloatTensor,
    CFloatTensor0D,
    CFloatTensor1D,
    CFloatTensor2D,
    CFloatTensor3D,
    CHalfTensor,
    CHalfTensor0D,
    CHalfTensor1D,
    CHalfTensor2D,
    CHalfTensor3D,
    CharTensor,
    CharTensor0D,
    CharTensor1D,
    CharTensor2D,
    CharTensor3D,
    ComplexFloatingTensor,
    ComplexFloatingTensor0D,
    ComplexFloatingTensor1D,
    ComplexFloatingTensor2D,
    ComplexFloatingTensor3D,
    DoubleTensor,
    DoubleTensor0D,
    DoubleTensor1D,
    DoubleTensor2D,
    DoubleTensor3D,
    FloatingTensor,
    FloatingTensor0D,
    FloatingTensor1D,
    FloatingTensor2D,
    FloatingTensor3D,
    FloatTensor,
    FloatTensor0D,
    FloatTensor1D,
    FloatTensor2D,
    FloatTensor3D,
    HalfTensor,
    HalfTensor0D,
    HalfTensor1D,
    HalfTensor2D,
    HalfTensor3D,
    IntTensor,
    IntTensor0D,
    IntTensor1D,
    IntTensor2D,
    IntTensor3D,
    LongTensor,
    LongTensor0D,
    LongTensor1D,
    LongTensor2D,
    LongTensor3D,
    ShortTensor,
    ShortTensor0D,
    ShortTensor1D,
    ShortTensor2D,
    ShortTensor3D,
    SignedIntegerTensor,
    SignedIntegerTensor0D,
    SignedIntegerTensor1D,
    SignedIntegerTensor2D,
    SignedIntegerTensor3D,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
    UnsignedIntegerTensor,
    UnsignedIntegerTensor0D,
    UnsignedIntegerTensor1D,
    UnsignedIntegerTensor2D,
    UnsignedIntegerTensor3D,
)

r"""/!\ The following type hints are meant for type annotation only, not for runtime checks.
"""

NumberLike = Union[BuiltinNumber, NumpyNumberLike, Tensor0D]
ScalarLike = Union[BuiltinScalar, NumpyScalarLike, Tensor0D]
TensorLike = Union[torch.Tensor, np.ndarray]

T_TensorLike = TypeVar(
    "T_TensorLike",
    bound=Union[torch.Tensor, np.ndarray],
    covariant=True,
)

del np
