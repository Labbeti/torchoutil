#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import struct
from numbers import Real
from typing import Optional, TypeVar

from .functools import function_alias

T = TypeVar("T", bound=Real)


def clip(x: T, xmin: Optional[T] = None, xmax: Optional[T] = None) -> T:
    if xmin is not None:
        x = max(x, xmin)
    if xmax is not None:
        x = min(x, xmax)
    return x


@function_alias(clip)
def clamp(*args, **kwargs):
    ...


def nextdown(x: float) -> float:
    return -_nextup(-x)


def nextafter(x: float, y: float) -> float:
    """Equivalent to `math.nextafter` for python <=3.8."""

    # BASED on https://stackoverflow.com/questions/10420848/how-do-you-get-the-next-value-in-the-floating-point-sequence/10426033#10426033
    # If either argument is a NaN, return that argument.
    # This matches the implementation in decimal.Decimal
    if math.isnan(x):
        return x
    if math.isnan(y):
        return y

    if y == x:
        return y
    elif y > x:
        return _nextup(x)
    else:
        return nextdown(x)


def _nextup(x: float) -> float:
    # NaNs and positive infinity map to themselves.
    if math.isnan(x) or (math.isinf(x) and x > 0):
        return x

    # 0.0 and -0.0 both map to the smallest +ve float.
    if x == 0.0:
        x = 0.0

    n = struct.unpack("<q", struct.pack("<d", x))[0]
    if n >= 0:
        n += 1
    else:
        n -= 1
    return struct.unpack("<d", struct.pack("<q", n))[0]
