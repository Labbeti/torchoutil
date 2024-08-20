#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
from typing import Any, Callable, Generic, List, TypeVar

T = TypeVar("T")


def identity(x: T) -> T:
    """Identity function placeholder."""
    return x


class Compose(Generic[T]):
    def __init__(self, *fns: Callable[[T], T]) -> None:
        super().__init__()
        self.fns = fns

    def __call__(self, x: T) -> T:
        for fn in self.fns:
            x = fn(x)
        return x


def compose(*fns: Callable[[T], T]) -> Callable[[T], T]:
    return Compose(*fns)


def get_argnames(fn: Callable) -> List[str]:
    """Get arguments names of a method, function or callable object."""
    if inspect.ismethod(fn):
        # If method, remove 'self' arg
        argnames = fn.__code__.co_varnames[1:]  # type: ignore
    elif inspect.isfunction(fn):
        argnames = fn.__code__.co_varnames
    else:
        argnames = fn.__call__.__code__.co_varnames

    argnames = list(argnames)
    return argnames


def filter_and_call(fn: Callable, **kwargs: Any) -> Any:
    """Filter kwargs with function arg names and call function."""
    argnames = get_argnames(fn)
    kwargs_filtered = {
        name: value for name, value in kwargs.items() if name in argnames
    }
    return fn(**kwargs_filtered)
