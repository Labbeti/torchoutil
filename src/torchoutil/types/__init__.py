#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.types import *

from pyoutil.typing import (
    BuiltinNumber,
    DataclassInstance,
    NamedTupleInstance,
    SizedIterable,
    is_builtin_number,
    is_dataclass_instance,
    is_dict_str,
    is_iterable_bool,
    is_iterable_bytes_or_list,
    is_iterable_float,
    is_iterable_int,
    is_iterable_iterable_int,
    is_iterable_str,
    is_list_bool,
    is_list_float,
    is_list_int,
    is_list_list_str,
    is_list_number,
    is_list_str,
    is_mapping_str,
    is_namedtuple_instance,
    is_sequence_bool,
    is_sequence_int,
    is_sequence_str,
    is_tuple_str,
)

from .classes import ACCEPTED_NUMPY_DTYPES, TORCH_DTYPES, np, numpy
from .guards import (
    is_bool_tensor,
    is_integer_dtype,
    is_integer_tensor,
    is_iterable_tensor,
    is_list_tensor,
    is_numpy_number_like,
    is_scalar,
    is_tensor0d,
    is_tuple_tensor,
)
