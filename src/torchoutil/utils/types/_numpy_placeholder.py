#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module used when numpy is not installed."""

from typing import Any, Type

from torch import Tensor
from typing_extensions import Never


class number(Tensor):
    ...


class ndarray(Tensor):
    ...


class dtype(Type[None]):
    ...


def asarray(x: Any, *args, **kwargs) -> Never:
    raise NotImplementedError(
        "Cannot call function 'asarray'. Please install numpy first."
    )
