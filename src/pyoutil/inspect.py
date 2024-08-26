#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
from typing import Any, Protocol, runtime_checkable


def get_current_fn_name() -> str:
    try:
        return inspect.currentframe().f_back.f_code.co_name  # type: ignore
    except AttributeError:
        return ""


@runtime_checkable
class ClassLike(Protocol):
    __module__: str
    __qualname__: str


def get_fullname(obj: Any, *, inst_suffix: str = "(...)") -> str:
    """Returns the classname of an object with parent modules.

    Args:
        obj: Object to scan.
        inst_suffix: Suffix appended to the classname in case the object is an instance of a class.

    Example 1
    ----------
    >>> get_fullname([0, 1, 2])
    ... 'builtins.list(...)'
    >>> get_fullname(1.0)
    ... 'builtins.float(...)'
    >>> class A: def f(self): return 0
    >>> a = A()
    >>> get_fullname(a)
    ... '__main__.A(...)'
    >>> get_fullname(A)
    ... '__main__.A'
    >>> get_fullname(a.f)
    ... '__main__.A.f'
    >>> get_fullname(A.f)
    ... '__main__.A.f'
    """
    if isinstance(obj, ClassLike):
        name = f"{obj.__module__}.{obj.__qualname__}"
    elif inspect.ismodule(obj):
        name = obj.__name__
    else:
        cls = obj.__class__
        name = f"{cls.__module__}.{cls.__qualname__}{inst_suffix}"
    return name
