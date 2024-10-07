#!/usr/bin/env python
# -*- coding: utf-8 -*-

from difflib import SequenceMatcher
from typing import Callable, Sequence

from .collections import argmax


def sequence_matcher_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def find_closest_in_list(
    x: str,
    lst: Sequence[str],
    sim_fn: Callable[[str, str], float] = sequence_matcher_ratio,
) -> str:
    if len(lst) <= 0:
        msg = f"Invalid argument {lst=}. (expected non-empty sequence of strings)"
        raise ValueError(msg)

    ratios = [sim_fn(x, expected_i) for expected_i in lst]
    idx = argmax(ratios)
    closest = lst[idx]

    return closest
