#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Union, overload

from torchoutil.utils.collections import dict_list_to_list_dict, list_dict_to_dict_list
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
    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            fpath.parent.mkdir(parents=True, exist_ok=True)

    if to_builtins:
        data = to_builtin(data)

    if isinstance(data, Mapping):
        data = dict_list_to_list_dict(data)

    if len(data) <= 0:
        raise ValueError(f"Invalid argument {data=}. (found empty iterable)")

    if header:
        writer_cls = csv.DictWriter
    else:
        writer_cls = csv.writer

    fieldnames = list(data[0].keys())  # type: ignore
    file = io.StringIO()
    writer = writer_cls(file, fieldnames, **csv_writer_kwargs)
    if header:
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
    **csv_reader_kwargs,
) -> Dict[str, List[Any]]:
    ...


@overload
def load_csv(
    fpath: Union[str, Path],
    /,
    *,
    orient: Literal["list", "dict"] = "list",
    header: bool = True,
    **csv_reader_kwargs,
) -> List[Dict[str, Any]]:
    ...


def load_csv(
    fpath: Union[str, Path],
    /,
    *,
    orient: Literal["list", "dict"] = "list",
    header: bool = True,
    **csv_reader_kwargs,
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    if header:
        reader_cls = csv.DictReader
    else:
        reader_cls = csv.reader

    with open(fpath, "r") as file:
        reader = reader_cls(file, **csv_reader_kwargs)
        data = list(reader)

    if not header:
        data = [
            {f"{j}": data_ij for j, data_ij in enumerate(data_i)} for data_i in data
        ]

    if orient == "dict":
        data = list_dict_to_dict_list(data, key_mode="same")
    elif orient == "list":
        pass
    else:
        raise ValueError(
            f"Invalid argument {orient=}. (expected one of {ORIENT_VALUES})"
        )

    return data
