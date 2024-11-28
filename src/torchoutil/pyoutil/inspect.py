#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
from typing import Any, get_args


def get_current_fn_name() -> str:
    try:
        return inspect.currentframe().f_back.f_code.co_name  # type: ignore
    except AttributeError:
        return ""


def get_fullname(x: Any, *, inst_suffix: str = "(...)") -> str:
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
    if hasattr(x, "__module__") and hasattr(x, "__qualname__"):
        name = f"{x.__module__}.{x.__qualname__}"
    elif inspect.ismodule(x):
        name = x.__name__
    else:
        cls = x.__class__
        name = f"{cls.__module__}.{cls.__qualname__}{inst_suffix}"

    cls_args = get_args(x)
    if len(cls_args) != 0:
        argsnames = [get_fullname(arg, inst_suffix=inst_suffix) for arg in cls_args]
        argsnames_str = ", ".join(argsnames)
        name = f"{name}[{argsnames_str}]"

    return name
