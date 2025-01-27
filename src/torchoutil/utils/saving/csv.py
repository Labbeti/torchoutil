#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from torchoutil.core.packaging import _PANDAS_AVAILABLE
from torchoutil.pyoutil.csv import ORIENT_VALUES, Orient, _setup_path
from torchoutil.pyoutil.csv import load_csv as load_csv_base
from torchoutil.pyoutil.csv import to_csv as to_csv_base
from torchoutil.utils.saving.common import to_builtin

if _PANDAS_AVAILABLE:
    import pandas as pd

CSVBackend = Literal["csv", "pandas"]
CSV_BACKENDS = ("csv", "pandas")


@overload
def load_csv(
    fpath: Union[str, Path],
    /,
    *,
    orient: Literal["dict"],
    header: bool = True,
    comment_start: Optional[str] = None,
    strip_content: bool = False,
    backend: CSVBackend = "csv",
    # CSV reader kwargs
    delimiter: Optional[str] = None,
    **csv_reader_kwds,
) -> Dict[str, List[Any]]:
    ...


@overload
def load_csv(
    fpath: Union[str, Path],
    /,
    *,
    orient: Literal["list"] = "list",
    header: bool = True,
    comment_start: Optional[str] = None,
    strip_content: bool = False,
    backend: CSVBackend = "csv",
    # CSV reader kwargs
    delimiter: Optional[str] = None,
    **csv_reader_kwds,
) -> List[Dict[str, Any]]:
    ...


def load_csv(
    fpath: Union[str, Path],
    /,
    *,
    orient: Orient = "list",
    header: bool = True,
    comment_start: Optional[str] = None,
    strip_content: bool = False,
    backend: CSVBackend = "csv",
    # CSV reader kwargs
    delimiter: Optional[str] = None,
    **csv_reader_kwds,
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    if backend == "csv":
        return load_csv_base(
            fpath,
            orient=orient,
            header=header,
            comment_start=comment_start,
            strip_content=strip_content,
            delimiter=delimiter,
            **csv_reader_kwds,
        )

    elif backend == "pandas":
        return _load_csv_with_pandas(
            fpath,
            orient=orient,
            header=header,
            comment_start=comment_start,
            strip_content=strip_content,
            delimiter=delimiter,
            **csv_reader_kwds,
        )

    else:
        msg = f"Invalid argument {backend=}. (expected one of {CSV_BACKENDS})"
        raise ValueError(msg)


def _load_csv_with_pandas(
    fpath: Union[str, Path],
    /,
    *,
    orient: Orient = "list",
    header: bool = True,
    comment_start: Optional[str] = None,
    strip_content: bool = False,
    # CSV reader kwargs
    delimiter: Optional[str] = None,
    **csv_reader_kwds,
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    backend = "pandas"

    if not _PANDAS_AVAILABLE:
        msg = f"Invalid argument {backend=} without pandas installed."
        raise ValueError(msg)

    if strip_content:
        msg = f"Invalid argument {strip_content=} with {backend=}."
        raise ValueError(msg)

    if comment_start is not None:
        msg = f"Invalid argument {comment_start=} with {backend=}."
        raise ValueError(msg)

    if len(csv_reader_kwds) > 0:
        msg = f"Invalid arguments {csv_reader_kwds=} with {backend=}."
        raise ValueError(msg)

    df = pd.read_csv(fpath, delimiter=delimiter)  # type: ignore

    if orient == "list":
        return df.to_dict("records")  # type: ignore
    elif orient == "dict":
        return df.to_dict("list")  # type: ignore
    else:
        msg = f"Invalid argument {orient=}. (expected one of {ORIENT_VALUES})"
        raise ValueError(msg)


def to_csv(
    data: Union[Iterable[Mapping[str, Any]], Mapping[str, Iterable[Any]]],
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    to_builtins: bool = False,
    make_parents: bool = True,
    backend: CSVBackend = "csv",
    header: bool = True,
    **csv_writer_kwds,
) -> str:
    """Dump content to csv format."""
    if to_builtins:
        data = to_builtin(data)

    if backend == "csv":
        return to_csv_base(
            data,
            fpath,
            overwrite=overwrite,
            make_parents=make_parents,
            header=header,
            **csv_writer_kwds,
        )

    elif backend == "pandas":
        return _to_csv_with_pandas(
            data,
            fpath,
            overwrite=overwrite,
            make_parents=make_parents,
        )

    else:
        msg = f"Invalid argument {backend=}. (expected one of {CSV_BACKENDS})"
        raise ValueError(msg)


def _to_csv_with_pandas(
    data: Union[Iterable[Mapping[str, Any]], Mapping[str, Iterable[Any]]],
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
) -> str:
    backend = "pandas"
    if not _PANDAS_AVAILABLE:
        msg = f"Invalid argument {backend=} without pandas installed."
        raise ValueError(msg)

    fpath = _setup_path(fpath, overwrite, make_parents)
    df = pd.DataFrame(data)  # type: ignore

    file = io.StringIO()
    df.to_csv(file)
    content = file.getvalue()
    file.close()

    if fpath is not None:
        fpath.write_text(content)

    return content
