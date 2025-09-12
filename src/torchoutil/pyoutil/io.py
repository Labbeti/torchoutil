#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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


def _setup_path(
    fpath: Union[str, Path, os.PathLike, None],
    overwrite: bool,
    make_parents: bool,
) -> Optional[Path]:
    """Resolve & expand path and create intermediate parents."""
    if not isinstance(fpath, (str, Path, os.PathLike)):
        return fpath

    fpath = Path(fpath).resolve().expanduser()
    if not overwrite and fpath.exists():
        raise FileExistsError(f"File {fpath} already exists.")
    elif make_parents:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    return fpath
