#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from torchoutil.pyoutil.typing import BuiltinNumber, BuiltinScalar
from torchoutil.types.classes import np
from torchoutil.types.tensor_typing import (  # noqa: F401
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
    CFloatTensor0D,
    CFloatTensor1D,
    CFloatTensor2D,
    CFloatTensor3D,
    CharTensor,
    CharTensor0D,
    CharTensor1D,
    CharTensor2D,
    CharTensor3D,
    DoubleTensor,
    DoubleTensor0D,
    DoubleTensor1D,
    DoubleTensor2D,
    DoubleTensor3D,
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
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
)

r"""/!\ The following type hints meants for type annotation only, not for runtime checks.
"""

NumpyNumberLike = Union[np.ndarray, np.number]
NumpyScalarLike = Union[np.ndarray, np.generic]

NumberLike = Union[BuiltinNumber, NumpyNumberLike, Tensor0D]
ScalarLike = Union[BuiltinScalar, NumpyScalarLike, Tensor0D]
