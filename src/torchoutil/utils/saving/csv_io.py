#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import io
from csv import DictReader, DictWriter
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

from pyoutil.collections import dict_list_to_list_dict, list_dict_to_dict_list
from torchoutil.utils.saving.common import to_builtin

ORIENT_VALUES = ("list", "dict")


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
    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            fpath.parent.mkdir(parents=True, exist_ok=True)

    if to_builtins:
        data = to_builtin(data)

    if isinstance(data, Mapping):
        data = dict_list_to_list_dict(data)  # type: ignore
    else:
        data = list(data)

    if header:
        writer_cls = DictWriter
        if len(data) == 0:
            fieldnames = []
        else:
            fieldnames = [str(k) for k in data[0].keys()]
        csv_writer_kwargs["fieldnames"] = fieldnames
    else:
        writer_cls = csv.writer

    file = io.StringIO()
    writer = writer_cls(file, **csv_writer_kwargs)
    if isinstance(writer, DictWriter):
        writer.writeheader()
    writer.writerows(data)  # type: ignore
    content = file.getvalue()
    file.close()

    if fpath is not None:
        fpath.write_text(content)

    return content


@overload
def load_csv(
    fpath: Union[str, Path],
    /,
    *,
    orient: Literal["dict"],
    header: bool = True,
    comment_start: Optional[str] = None,
    **csv_reader_kwargs,
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
    **csv_reader_kwargs,
) -> List[Dict[str, Any]]:
    ...


def load_csv(
    fpath: Union[str, Path],
    /,
    *,
    orient: Literal["list", "dict"] = "list",
    header: bool = True,
    comment_start: Optional[str] = None,
    **csv_reader_kwargs,
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Load content from csv filepath."""
    if header:
        reader_cls = DictReader
    else:
        reader_cls = csv.reader

    with open(fpath, "r") as file:
        reader = reader_cls(file, **csv_reader_kwargs)
        data = list(reader)

        if comment_start is None:
            pass
        elif header:
            data = [
                line
                for line in data
                if not next(iter(line.values())).startswith(comment_start)
            ]
        else:
            data = [line for line in data if not line[0].startswith(comment_start)]

    if not header:
        data = [
            {str(j): data_ij for j, data_ij in enumerate(data_i)} for data_i in data
        ]

    if orient == "dict":
        data = list_dict_to_dict_list(data, key_mode="same")  # type: ignore
    elif orient == "list":
        pass
    else:
        raise ValueError(
            f"Invalid argument {orient=}. (expected one of {ORIENT_VALUES})"
        )

    return data  # type: ignore
