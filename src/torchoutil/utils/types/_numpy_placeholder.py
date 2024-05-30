#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module used when numpy is not installed."""

from typing import Any, NewType, Type

from torch import Tensor
from typing_extensions import Never

number = NewType("number", Tensor)
ndarray = NewType("ndarray", Tensor)
dtype = NewType("dtype", Type[None])


def asarray(x: Any, *args, **kwargs) -> Never:
    raise NotImplementedError(
        "Cannot call function 'asarray'. Please install numpy first."
    )
