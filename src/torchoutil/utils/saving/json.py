#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Optional, Union

from torchoutil.pyoutil.json import load_json  # noqa: F401
from torchoutil.pyoutil.json import to_json as to_json_base
from torchoutil.utils.saving.common import to_builtin


def to_json(
    data: Any,
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    to_builtins: bool = False,
    # JSON dump kwargs
    indent: Optional[int] = 4,
    ensure_ascii: bool = False,
    **json_dump_kwds,
) -> str:
    """Dump content to csv format."""
    if to_builtins:
        data = to_builtin(data)

    return to_json_base(
        data,
        fpath,
        overwrite=overwrite,
        make_parents=make_parents,
        indent=indent,
        ensure_ascii=ensure_ascii,
        **json_dump_kwds,
    )
