#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor
from torch.types import *

from torchoutil.core.dtype_enum import DTypeEnum
from torchoutil.core.make import DeviceLike, DTypeLike, GeneratorLike
from torchoutil.pyoutil.typing import *

from ._typing import T_TensorOrArray, TensorOrArray
from .guards import (
    is_bool_tensor,
    is_bool_tensor1d,
    is_builtin_number,
    is_builtin_scalar,
    is_complex_tensor,
    is_dict_str_tensor,
    is_floating_tensor,
    is_integral_dtype,
    is_integral_tensor,
    is_integral_tensor1d,
    is_iterable_tensor,
    is_list_tensor,
    is_number_like,
    is_numpy_number_like,
    is_numpy_scalar_like,
    is_scalar_like,
    is_tensor0d,
    is_tensor_like,
    is_tensor_or_array,
    is_tuple_tensor,
    isinstance_guard,
)
from .tensor_subclasses import (
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
)
