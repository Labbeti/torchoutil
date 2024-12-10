#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Union

from torchoutil.pyoutil.pickle import load_pickle  # noqa: F401
from torchoutil.pyoutil.pickle import to_pickle as to_pickle_base
from torchoutil.utils.saving.common import to_builtin


def to_pickle(obj: Any, fpath: Union[str, Path], *, to_builtins: bool = False) -> bytes:
    """Dump content to PICKLE format into a bytes and/or file."""
    if to_builtins:
        obj = to_builtin(obj)
    return to_pickle_base(obj, fpath)
