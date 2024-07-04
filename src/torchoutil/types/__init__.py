#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.utils.stdlib.typing import (
    BuiltinScalar,
    is_builtin_scalar,
    is_dataclass_instance,
    is_dict_str,
    is_iterable_bool,
    is_iterable_bytes_or_list,
    is_iterable_int,
    is_iterable_iterable_int,
    is_iterable_str,
    is_list_bool,
    is_list_int,
    is_list_list_str,
    is_list_str,
    is_mapping_str,
    is_namedtuple_instance,
    is_sequence_bool,
    is_sequence_int,
    is_sequence_str,
    is_tuple_str,
)

from .classes import (
    ACCEPTED_NUMPY_DTYPES,
    TORCH_DTYPES,
    BoolTensor0D,
    BoolTensor1D,
    BoolTensor2D,
    BoolTensor3D,
    BoolTensor4D,
    BoolTensor5D,
    ByteTensor0D,
    ByteTensor1D,
    ByteTensor2D,
    ByteTensor3D,
    ByteTensor4D,
    ByteTensor5D,
    CharTensor0D,
    CharTensor1D,
    CharTensor2D,
    CharTensor3D,
    CharTensor4D,
    CharTensor5D,
    DoubleTensor0D,
    DoubleTensor1D,
    DoubleTensor2D,
    DoubleTensor3D,
    DoubleTensor4D,
    DoubleTensor5D,
    FloatTensor0D,
    FloatTensor1D,
    FloatTensor2D,
    FloatTensor3D,
    FloatTensor4D,
    FloatTensor5D,
    HalfTensor0D,
    HalfTensor1D,
    HalfTensor2D,
    HalfTensor3D,
    HalfTensor4D,
    HalfTensor5D,
    IntTensor0D,
    IntTensor1D,
    IntTensor2D,
    IntTensor3D,
    IntTensor4D,
    IntTensor5D,
    LongTensor0D,
    LongTensor1D,
    LongTensor2D,
    LongTensor3D,
    LongTensor4D,
    LongTensor5D,
    NumpyScalar,
    Scalar,
    ShortTensor0D,
    ShortTensor1D,
    ShortTensor2D,
    ShortTensor3D,
    ShortTensor4D,
    ShortTensor5D,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
    Tensor4D,
    Tensor5D,
    np,
    numpy,
)
from .guards import (
    is_iterable_tensor,
    is_list_tensor,
    is_numpy_scalar,
    is_scalar,
    is_torch_scalar,
    is_tuple_tensor,
)
