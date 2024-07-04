#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from re import Pattern
from typing import Callable, Iterable, List, TypeVar, Union

T = TypeVar("T")

PatternLike = Union[str, Pattern]


def compile_patterns(
    patterns: Union[PatternLike, Iterable[PatternLike]]
) -> List[Pattern]:
    """Compile patterns-like to a list."""
    if isinstance(patterns, (str, Pattern)):
        patterns = [patterns]
    patterns = [re.compile(pattern) for pattern in patterns]
    return patterns


def find_pattern(
    x: str,
    patterns: List[Pattern],
    *,
    match_fn: Callable[[Pattern, str], bool] = re.search,  # type: ignore
    default: T = -1,
) -> Union[int, T]:
    """Find index of a pattern that match the first argument. If no pattern matches, returns the default value (-1)."""
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
    """Returns True if at least 1 pattern match the first argument."""
    return find_pattern(x, patterns, match_fn=match_fn, default=-1) != -1
