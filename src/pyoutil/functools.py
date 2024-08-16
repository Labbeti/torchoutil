#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TypeVar

T = TypeVar("T")


def identity(x: T) -> T:
    """Identity function placeholder."""
    return x
