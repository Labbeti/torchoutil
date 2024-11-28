#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
from typing import Any, Callable, Generic, List, TypeVar, overload

T = TypeVar("T")
U = TypeVar("U")


def identity(x: T) -> T:
    """Identity function placeholder."""
    return x


class Compose(Generic[T, U]):
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(
        self,
        fn0: Callable[[T], U],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        fn0: Callable[[T], Any],
        fn1: Callable[[Any], U],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        fn0: Callable[[T], Any],
        fn1: Callable[[Any], Any],
        fn2: Callable[[Any], U],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        fn0: Callable[[T], Any],
        fn1: Callable[[Any], Any],
        fn2: Callable[[Any], Any],
        fn3: Callable[[Any], U],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        fn0: Callable[[T], Any],
        fn1: Callable[[Any], Any],
        fn2: Callable[[Any], Any],
        fn3: Callable[[Any], Any],
        fn4: Callable[[Any], U],
        /,
    ) -> None:
        ...

    @overload
    def __init__(self, *fns: Callable) -> None:
        ...

    def __init__(self, *fns: Callable) -> None:
        super().__init__()
        self.fns = fns

    def __call__(self, x: T) -> U:
        for fn in self.fns:
            x = fn(x)
        return x  # type: ignore

    def __getitem__(self, idx: int, /) -> Callable[[Any], Any]:
        return self.fns[idx]

    def __len__(self) -> int:
        return len(self.fns)


compose = Compose  # type: ignore


def get_argnames(fn: Callable) -> List[str]:
    """Get arguments names of a method, function or callable object."""
    if inspect.ismethod(fn):
        # If method, remove 'self' arg
        argnames = fn.__code__.co_varnames[1:]  # type: ignore
    elif inspect.isfunction(fn):
        argnames = fn.__code__.co_varnames
    else:
        argnames = fn.__call__.__code__.co_varnames  # type: ignore

    argnames = list(argnames)
    return argnames


def filter_and_call(fn: Callable[..., T], **kwargs: Any) -> T:
    """Filter kwargs with function arg names and call function."""
    argnames = get_argnames(fn)
    kwargs_filtered = {
        name: value for name, value in kwargs.items() if name in argnames
    }
    return fn(**kwargs_filtered)
