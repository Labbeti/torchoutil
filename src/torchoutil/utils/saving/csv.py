#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from torchoutil.pyoutil.csv import load_csv as load_csv_base
from torchoutil.pyoutil.csv import to_csv as to_csv_base
from torchoutil.utils.saving.common import to_builtin

if _PANDAS_AVAILABLE:
    import pandas as pd

CSVBackend = Literal["csv", "pandas"]
CSV_BACKENDS = ("csv", "pandas")
ORIENTS = ("list", "dict")


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
    **csv_reader_kwds,
) -> List[Dict[str, Any]]:
    ...


def load_csv(
    fpath: Union[str, Path],
    /,
    *,
    orient: Literal["list", "dict"] = "list",
    header: bool = True,
    comment_start: Optional[str] = None,
    strip_content: bool = False,
    backend: CSVBackend = "csv",
    **csv_reader_kwds,
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    if backend == "csv":
        return load_csv_base(
            fpath,
            orient=orient,
            header=header,
            comment_start=comment_start,
            strip_content=strip_content,
            **csv_reader_kwds,
        )

    elif backend == "pandas":
        return _load_csv_with_pandas(
            fpath,
            orient=orient,
            header=header,
            comment_start=comment_start,
            strip_content=strip_content,
            **csv_reader_kwds,
        )

    else:
        msg = f"Invalid argument {backend=}. (expected one of {CSV_BACKENDS})"
        raise ValueError(msg)


def _load_csv_with_pandas(
    fpath: Union[str, Path],
    /,
    *,
    orient: Literal["list", "dict"] = "list",
    header: bool = True,
    comment_start: Optional[str] = None,
    strip_content: bool = False,
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

    df = pd.read_csv(fpath)  # type: ignore

    if orient == "list":
        return df.to_dict("records")  # type: ignore
    elif orient == "dict":
        return df.to_dict("list")  # type: ignore
    else:
        msg = f"Invalid argument {orient=}. (expected one of {ORIENTS})"
        raise ValueError(msg)


def to_csv(
    data: Union[Iterable[Mapping[str, Any]], Mapping[str, Iterable[Any]]],
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    to_builtins: bool = False,
    make_parents: bool = True,
    header: bool = True,
    **csv_writer_kwds,
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
        **csv_writer_kwds,
    )
