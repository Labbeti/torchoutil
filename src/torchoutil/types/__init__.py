#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.types import *

from pyoutil.typing import *

from .classes import ACCEPTED_NUMPY_DTYPES, TORCH_DTYPES, np, numpy
from .guards import (
    is_bool_tensor,
    is_bool_tensor1d,
    is_builtin_number,
    is_builtin_scalar,
    is_integer_dtype,
    is_integer_tensor,
    is_integer_tensor1d,
    is_iterable_tensor,
    is_list_tensor,
    is_number_like,
    is_numpy_number_like,
    is_numpy_scalar_like,
    is_scalar_like,
    is_tensor0d,
    is_tuple_tensor,
)
