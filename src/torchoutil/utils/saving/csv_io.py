#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Iterable, Mapping, Union

from pyoutil.csv import load_csv  # noqa: F401
from pyoutil.csv import to_csv as to_csv_base
from torchoutil.utils.saving.common import to_builtin


def to_csv(
    data: Union[Iterable[Mapping[str, Any]], Mapping[str, Iterable[Any]]],
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    to_builtins: bool = False,
    make_parents: bool = True,
    header: bool = True,
    **csv_writer_kwargs,
) -> str:
    """Dump content to csv format."""
    if to_builtins:
        data = to_builtin(data)

    return to_csv_base(
        data,
        fpath,
        overwrite=overwrite,
        make_parents=make_parents,
        header=header,
        **csv_writer_kwargs,
    )
