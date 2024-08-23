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
    TypeVar,
    Union,
    overload,
)

from pyoutil.collections import dict_list_to_list_dict, list_dict_to_dict_list

T = TypeVar("T")

ORIENT_VALUES = ("list", "dict")


def to_csv(
    data: Union[Iterable[Mapping[str, Any]], Mapping[str, Iterable[Any]]],
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    header: bool = True,
    align_content: bool = False,
    **csv_writer_kwargs,
) -> str:
    """Dump content to csv format."""
    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            fpath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, Mapping):
        data_lst = dict_list_to_list_dict(data)  # type: ignore
    else:
        data_lst = list(data)
    del data

    if header:
        writer_cls = DictWriter
        if len(data_lst) == 0:
            fieldnames = []
        else:
            fieldnames = [str(k) for k in data_lst[0].keys()]
    else:
        writer_cls = csv.writer
        fieldnames = list(range(len(next(data_lst))))

    if align_content:
        old_fieldnames = fieldnames
        data_lst = _stringify(data_lst)
        fieldnames = _stringify(fieldnames)
        max_num_chars = {
            k: max(max(len(data_i[k]) for data_i in data_lst), len(k)) + 1
            for k in fieldnames
        }

        fieldnames = [f"{{:^{max_num_chars[k]}s}}".format(k) for k in fieldnames]
        old_to_new_fieldnames = dict(zip(old_fieldnames, fieldnames))

        data_lst = [
            {
                old_to_new_fieldnames[k]: f"{{:^{max_num_chars[k]}s}}".format(v)
                for k, v in data_i.items()
            }
            for data_i in data_lst
        ]

    if header:
        csv_writer_kwargs["fieldnames"] = fieldnames

    file = io.StringIO()
    writer = writer_cls(file, **csv_writer_kwargs)
    if isinstance(writer, DictWriter):
        writer.writeheader()
    writer.writerows(data_lst)  # type: ignore
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
    strip_content: bool = False,
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
    strip_content: bool = False,
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
    strip_content: bool = False,
    **csv_reader_kwargs,
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Load content from csv filepath."""
    if header:
        reader_cls = DictReader
    else:
        reader_cls = csv.reader

    with open(fpath, "r") as file:
        reader = reader_cls(file, **csv_reader_kwargs)
        data_lst = list(reader)

        if comment_start is None:
            pass
        elif header:
            data_lst = [
                line
                for line in data_lst
                if not next(iter(line.values())).startswith(comment_start)
            ]
        else:
            data_lst = [
                line for line in data_lst if not line[0].startswith(comment_start)
            ]

    if not header:
        data_lst = [
            {str(j): data_ij for j, data_ij in enumerate(data_i)} for data_i in data_lst
        ]

    if strip_content:
        data_lst = [
            {k.strip(): v.strip() for k, v in data_i.items()} for data_i in data_lst
        ]

    if orient == "dict":
        result = list_dict_to_dict_list(data_lst, key_mode="same")  # type: ignore
    elif orient == "list":
        result = data_lst
    else:
        raise ValueError(
            f"Invalid argument {orient=}. (expected one of {ORIENT_VALUES})"
        )

    return result  # type: ignore


def _stringify(x: T) -> T:
    if isinstance(x, str):
        return x
    elif isinstance(x, dict):
        return {_stringify(k): _stringify(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple, set, frozenset)):
        return type(x)(_stringify(xi) for xi in x)
    else:
        return str(x)
