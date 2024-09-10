#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

T = TypeVar("T", covariant=True)


def open_close_wrap(
    fn: Callable[..., T],
    fpath: Union[str, Path],
    open_kwds: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> T:
    if open_kwds is None:
        open_kwds = {}

    with open(fpath, **open_kwds) as file:
        return fn(file, *args, **kwargs)  # type: ignore
