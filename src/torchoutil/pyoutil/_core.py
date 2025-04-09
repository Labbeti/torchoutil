#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")
U = TypeVar("U")


def return_none(*args, **kwargs) -> None:
    """Return None function placeholder."""
    return None


def _decorator_factory(
    inner_fn: Optional[Callable[P, U]],
    *,
    pre_fn: Callable[..., Any] = return_none,
    post_fn: Callable[..., Any] = return_none,
) -> Callable[..., Callable[P, U]]:
    """Deprecated decorator for function aliases."""

    def wrapper_factory(fn: Callable[P, U]) -> Callable[P, U]:
        if inner_fn is None:
            _inner_fn = fn
        else:
            _inner_fn = inner_fn

        @wraps(_inner_fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> U:
            pre_fn(fn, *args, **kwargs)
            result = _inner_fn(*args, **kwargs)
            post_fn(fn, *args, **kwargs)
            return result

        return wrapped

    return wrapper_factory
