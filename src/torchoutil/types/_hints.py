#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from torch import (
    BoolTensor,
    ByteTensor,
    CharTensor,
    DoubleTensor,
    FloatTensor,
    HalfTensor,
    IntTensor,
    LongTensor,
    ShortTensor,
    Tensor,
)
from typing_extensions import Annotated

from pyoutil.typing import BuiltinNumber, BuiltinScalar
from torchoutil.types.classes import np

Tensor0D = Annotated[Tensor, "0D"]
Tensor1D = Annotated[Tensor, "1D"]
Tensor2D = Annotated[Tensor, "2D"]
Tensor3D = Annotated[Tensor, "3D"]
Tensor4D = Annotated[Tensor, "4D"]
Tensor5D = Annotated[Tensor, "5D"]

BoolTensor0D = Annotated[BoolTensor, "0D"]
BoolTensor1D = Annotated[BoolTensor, "1D"]
BoolTensor2D = Annotated[BoolTensor, "2D"]
BoolTensor3D = Annotated[BoolTensor, "3D"]
BoolTensor4D = Annotated[BoolTensor, "4D"]
BoolTensor5D = Annotated[BoolTensor, "5D"]

ByteTensor0D = Annotated[ByteTensor, "0D"]
ByteTensor1D = Annotated[ByteTensor, "1D"]
ByteTensor2D = Annotated[ByteTensor, "2D"]
ByteTensor3D = Annotated[ByteTensor, "3D"]
ByteTensor4D = Annotated[ByteTensor, "4D"]
ByteTensor5D = Annotated[ByteTensor, "5D"]

CharTensor0D = Annotated[CharTensor, "0D"]
CharTensor1D = Annotated[CharTensor, "1D"]
CharTensor2D = Annotated[CharTensor, "2D"]
CharTensor3D = Annotated[CharTensor, "3D"]
CharTensor4D = Annotated[CharTensor, "4D"]
CharTensor5D = Annotated[CharTensor, "5D"]

DoubleTensor0D = Annotated[DoubleTensor, "0D"]
DoubleTensor1D = Annotated[DoubleTensor, "1D"]
DoubleTensor2D = Annotated[DoubleTensor, "2D"]
DoubleTensor3D = Annotated[DoubleTensor, "3D"]
DoubleTensor4D = Annotated[DoubleTensor, "4D"]
DoubleTensor5D = Annotated[DoubleTensor, "5D"]

FloatTensor0D = Annotated[FloatTensor, "0D"]
FloatTensor1D = Annotated[FloatTensor, "1D"]
FloatTensor2D = Annotated[FloatTensor, "2D"]
FloatTensor3D = Annotated[FloatTensor, "3D"]
FloatTensor4D = Annotated[FloatTensor, "4D"]
FloatTensor5D = Annotated[FloatTensor, "5D"]

HalfTensor0D = Annotated[HalfTensor, "0D"]
HalfTensor1D = Annotated[HalfTensor, "1D"]
HalfTensor2D = Annotated[HalfTensor, "2D"]
HalfTensor3D = Annotated[HalfTensor, "3D"]
HalfTensor4D = Annotated[HalfTensor, "4D"]
HalfTensor5D = Annotated[HalfTensor, "5D"]

IntTensor0D = Annotated[IntTensor, "0D"]
IntTensor1D = Annotated[IntTensor, "1D"]
IntTensor2D = Annotated[IntTensor, "2D"]
IntTensor3D = Annotated[IntTensor, "3D"]
IntTensor4D = Annotated[IntTensor, "4D"]
IntTensor5D = Annotated[IntTensor, "5D"]

LongTensor0D = Annotated[LongTensor, "0D"]
LongTensor1D = Annotated[LongTensor, "1D"]
LongTensor2D = Annotated[LongTensor, "2D"]
LongTensor3D = Annotated[LongTensor, "3D"]
LongTensor4D = Annotated[LongTensor, "4D"]
LongTensor5D = Annotated[LongTensor, "5D"]

ShortTensor0D = Annotated[ShortTensor, "0D"]
ShortTensor1D = Annotated[ShortTensor, "1D"]
ShortTensor2D = Annotated[ShortTensor, "2D"]
ShortTensor3D = Annotated[ShortTensor, "3D"]
ShortTensor4D = Annotated[ShortTensor, "4D"]
ShortTensor5D = Annotated[ShortTensor, "5D"]


NumpyNumberLike = Union[np.ndarray, np.number]
NumpyScalarLike = Union[np.ndarray, np.generic]

NumberLike = Union[BuiltinNumber, NumpyNumberLike, Tensor0D]
ScalarLike = Union[BuiltinScalar, NumpyScalarLike, Tensor0D]
