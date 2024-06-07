#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchoutil.utils.stdlib.typing import (
    BuiltinScalar,
    is_builtin_scalar,
    is_dataclass_instance,
    is_dict_str,
    is_iterable_bool,
    is_iterable_bytes_list,
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
    NumpyScalar,
    Scalar,
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
