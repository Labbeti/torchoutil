#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import TypeVar

T = TypeVar("T")


def clip(x: T, xmin: T = -math.inf, xmax: T = math.inf) -> T:
    return min(max(x, xmin), xmax)


def clamp(x: T, xmin: T = -math.inf, xmax: T = math.inf) -> T:
    return clip(x, xmin, xmax)
