#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Any, Union

from torchoutil.pyoutil.pickle import dump_pickle as _dump_pickle_base
from torchoutil.pyoutil.pickle import load_pickle  # noqa: F401
from torchoutil.utils.saving.common import to_builtin


def dump_pickle(
    obj: Any,
    fpath: Union[str, Path, os.PathLike, None],
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    to_builtins: bool = False,
) -> bytes:
    """Dump content to PICKLE format into a bytes and/or file."""
    if to_builtins:
        obj = to_builtin(obj)
    return _dump_pickle_base(obj, fpath, overwrite=overwrite, make_parents=make_parents)


to_pickle = dump_pickle
