#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from re import Pattern
from typing import Callable, Iterable, List, Union

PatternLike = Union[str, Pattern]


def compile_patterns(
    patterns: Union[PatternLike, Iterable[PatternLike]]
) -> List[Pattern]:
    if isinstance(patterns, (str, Pattern)):
        patterns = [patterns]
    patterns = [re.compile(pattern) for pattern in patterns]
    return patterns


def find_pattern(
    x: str,
    patterns: List[Pattern],
    *,
    match_fn: Callable[[Pattern, str], bool] = re.search,  # type: ignore
    default: int = -1,
) -> int:
    for i, pattern in enumerate(patterns):
        if match_fn(pattern, x):
            return i
    return default


def pass_patterns(
    x: str,
    patterns: List[Pattern],
    *,
    match_fn: Callable[[Pattern, str], bool] = re.search,  # type: ignore
) -> bool:
    return find_pattern(x, patterns, match_fn=match_fn, default=-1) != -1
