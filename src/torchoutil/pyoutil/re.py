#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from functools import partial
from re import Pattern
from typing import Any, Callable, Iterable, List, TypeVar, Union

from torchoutil.pyoutil.collections import find
from torchoutil.pyoutil.inspect import get_current_fn_name

T = TypeVar("T")

PatternLike = Union[str, Pattern]
PatternListLike = Union[PatternLike, Iterable[PatternLike]]

pylog = logging.getLogger(__name__)


def compile_patterns(patterns: PatternListLike) -> List[Pattern]:
    """Compile patterns-like to a list."""
    if isinstance(patterns, (str, Pattern)):
        patterns = [patterns]
    patterns = [re.compile(pattern) for pattern in patterns]
    return patterns


def find_patterns(
    x: str,
    patterns: PatternListLike,
    *,
    match_fn: Callable[[PatternLike, str], Any] = re.search,
    default: T = -1,
) -> Union[int, T]:
    """Find index of a pattern that match the first argument. If no pattern matches, returns the default value (-1)."""
    patterns = compile_patterns(patterns)
    index = find(x, patterns, match_fn=match_fn, order="right", default=default)
    return index


def pass_patterns(
    x: str,
    include: PatternListLike,
    *,
    exclude: PatternListLike = (),
    match_fn: Callable[[PatternLike, str], Any] = re.search,
) -> bool:
    pylog.warning(
        f"Deprecate function call '{get_current_fn_name()}'. Use 'contained_patterns' instead."
    )
    return match_any_patterns(x, include, exclude=exclude, match_fn=match_fn)


def match_any_patterns(
    x: str,
    include: PatternListLike,
    *,
    exclude: PatternListLike = (),
    match_fn: Callable[[PatternLike, str], Any] = re.search,
) -> bool:
    """Returns True if at least 1 pattern match the first argument."""
    return (
        find_patterns(x, include, match_fn=match_fn, default=-1) != -1
        and find_patterns(x, exclude, match_fn=match_fn, default=-1) == -1
    )


def get_key_fn(
    patterns: PatternListLike,
    *,
    match_fn: Callable[[PatternLike, str], Any] = re.search,
) -> Callable[[str], int]:
    """
    Usage:
    ```
    >>> lst = ["a", "abc", "aa", "abcd"]
    >>> patterns = ["^ab"]  # sort list with elements starting with 'ab' first
    >>> list(sorted(lst, key=get_key_fn(patterns)))
    ... ["abc", "abcd", "a", "aa"]
    ```
    """
    patterns = compile_patterns(patterns)
    key_fn = partial(
        find_patterns,
        patterns=patterns,
        match_fn=match_fn,
        default=len(patterns),
    )
    return key_fn  # type: ignore


def sort_with_patterns(
    x: Iterable[str],
    patterns: PatternListLike,
    *,
    match_fn: Callable[[PatternLike, str], Any] = re.search,
    reverse: bool = False,
) -> List[str]:
    key_fn = get_key_fn(patterns, match_fn=match_fn)
    x = sorted(x, key=key_fn, reverse=reverse)
    return x
