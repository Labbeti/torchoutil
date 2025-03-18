#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from functools import lru_cache
from typing import Any, Callable, Optional, Type, TypeVar, Union, overload

from typing_extensions import ParamSpec

from ._core import _alias

P = ParamSpec("P")
U = TypeVar("U")


@overload
def warn_once(
    message: str,
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 1,
    source: Any = None,
) -> None:
    ...


@overload
def warn_once(
    message: Warning,
    category: Any = None,
    stacklevel: int = 1,
    source: Any = None,
) -> None:
    ...


@lru_cache(maxsize=None)
def warn_once(
    message: Union[str, Warning],
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 1,
    source: Any = None,
) -> None:
    warnings.warn(message, category, stacklevel, source)


def deprecated_alias(
    alternative: Callable[P, U],
    msg_fmt: str = "Deprecated call to '{fn_name}', use '{alternative_name}' instead.",
    warn_fn: Callable[[str], Any] = warn_once,
) -> Callable[..., Callable[P, U]]:
    alternative_name = alternative.__name__ if alternative is not None else "None"

    def pre_fn(fn, *args, **kwargs):
        msg = msg_fmt.format(fn_name=fn.__name__, alternative_name=alternative_name)
        warn_fn(msg)

    return _alias(alternative, pre_fn=pre_fn)


def deprecated(
    msg_fmt: str = "Deprecated call to '{fn_name}'.",
    warn_fn: Callable[[str], Any] = warn_once,
) -> Callable[[Callable[P, U]], Callable[P, U]]:
    def pre_fn(fn, *args, **kwargs):
        msg = msg_fmt.format(fn_name=fn.__name__)
        warn_fn(msg)

    return _alias(None, pre_fn=pre_fn)
