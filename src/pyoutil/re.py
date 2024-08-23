#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from functools import partial
from re import Pattern
from typing import Callable, Iterable, List, TypeVar, Union

from pyoutil.collections import find
from pyoutil.inspect import get_current_fn_name

T = TypeVar("T")

PatternLike = Union[str, Pattern]

pylog = logging.getLogger(__name__)


def compile_patterns(
    patterns: Union[PatternLike, Iterable[PatternLike]]
) -> List[Pattern]:
    """Compile patterns-like to a list."""
    if isinstance(patterns, (str, Pattern)):
        patterns = [patterns]
    patterns = [re.compile(pattern) for pattern in patterns]
    return patterns


def find_patterns(
    x: str,
    patterns: Union[PatternLike, Iterable[PatternLike]],
    *,
    match_fn: Callable[[Pattern, str], bool] = re.search,  # type: ignore
    default: T = -1,
) -> Union[int, T]:
    """Find index of a pattern that match the first argument. If no pattern matches, returns the default value (-1)."""
    patterns = compile_patterns(patterns)
    index = find(x, patterns, match_fn=match_fn, order="right", default=default)
    return index


def pass_patterns(
    x: str,
    include: Union[PatternLike, Iterable[PatternLike]],
    *,
    exclude: Union[PatternLike, Iterable[PatternLike]] = (),
    match_fn: Callable[[Pattern, str], bool] = re.search,  # type: ignore
) -> bool:
    pylog.warning(
        f"Deprecate function call '{get_current_fn_name()}'. Use 'contained_patterns' instead."
    )
    return contained_patterns(x, include, exclude=exclude, match_fn=match_fn)


def contained_patterns(
    x: str,
    include: Union[PatternLike, Iterable[PatternLike]],
    *,
    exclude: Union[PatternLike, Iterable[PatternLike]] = (),
    match_fn: Callable[[Pattern, str], bool] = re.search,  # type: ignore
) -> bool:
    """Returns True if at least 1 pattern match the first argument."""
    return (
        find_patterns(x, include, match_fn=match_fn, default=-1) != -1
        and find_patterns(x, exclude, match_fn=match_fn, default=-1) == -1
    )


def get_key_fn(
    patterns: Union[PatternLike, Iterable[PatternLike]],
    *,
    match_fn: Callable[[Pattern, str], bool] = re.search,  # type: ignore
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
    patterns: Union[PatternLike, Iterable[PatternLike]],
    *,
    match_fn: Callable[[Pattern, str], bool] = re.search,  # type: ignore
    reverse: bool = False,
) -> List[str]:
    key_fn = get_key_fn(patterns, match_fn=match_fn)
    x = sorted(x, key=key_fn, reverse=reverse)
    return x
