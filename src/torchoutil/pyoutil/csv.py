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
    get_args,
    overload,
)

from .collections import dict_list_to_list_dict, list_dict_to_dict_list
from .io import _setup_path
from .typing import isinstance_guard

T = TypeVar("T")

Orient = Literal["list", "dict"]


def dump_csv(
    data: Union[Iterable[Mapping[str, Any]], Mapping[str, Iterable[Any]], Iterable],
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    header: Union[bool, Literal["auto"]] = "auto",
    align_content: bool = False,
    replace_newline_by: Optional[str] = "\\n",
    **csv_writer_kwds,
) -> str:
    """Dump content to CSV format."""
    fpath = _setup_path(fpath, overwrite, make_parents)

    if header == "auto":
        header = isinstance_guard(
            data, (Mapping[str, Iterable], Iterable[Mapping[str, Any]])
        )

    if isinstance_guard(data, Mapping[str, Iterable]):
        data_lst = dict_list_to_list_dict(data)  # type: ignore
    elif isinstance_guard(data, Iterable[Mapping[str, Any]]):
        data_lst = [dict(data_i.items()) for data_i in data]
    elif not header and isinstance_guard(data, Iterable[str]):
        data_lst = [{"0": data_i} for data_i in data]
    elif not header and isinstance_guard(data, Iterable[Iterable]):
        data_lst = [dict(zip(map(str, range(len(data_i))), data)) for data_i in data]
    elif not header and isinstance_guard(data, Iterable):
        data_lst = [{"0": data_i} for data_i in data]
    else:
        raise TypeError(f"Invalid argument type {type(data)} with {header=}.")
    del data

    if header:
        writer_cls = DictWriter
    else:
        writer_cls = csv.writer

    if len(data_lst) == 0:
        fieldnames = []
    else:
        fieldnames = [str(k) for k in data_lst[0].keys()]

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

    if replace_newline_by is not None:

        def _replace_newline(s):
            if not isinstance(s, str):
                return s
            else:
                return s.replace("\n", replace_newline_by)

        data_lst = [
            {_replace_newline(k): _replace_newline(v) for k, v in data_i.items()}
            for data_i in data_lst
        ]

    if header:
        csv_writer_kwds["fieldnames"] = fieldnames

    file = io.StringIO()
    writer = writer_cls(file, **csv_writer_kwds)
    if isinstance(writer, DictWriter):
        writer.writeheader()
        writer.writerows(data_lst)
    else:
        data_lst = [tuple(data_i.values()) for data_i in data_lst]
        writer.writerows(data_lst)
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
    # CSV reader kwargs
    delimiter: Optional[str] = None,
    **csv_reader_kwds,
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    """Load content from csv filepath."""
    fpath = Path(fpath)
    if delimiter is None or delimiter is ...:
        delimiter = "\t" if fpath.suffix == ".tsv" else ","

    if header:
        reader_cls = DictReader
    else:
        reader_cls = csv.reader

    with open(fpath, "r") as file:
        reader = reader_cls(file, delimiter=delimiter, **csv_reader_kwds)
        raw_data_lst = list(reader)

    data_lst: List[Dict[str, Any]]
    if header:
        data_lst = raw_data_lst  # type: ignore
    else:
        data_lst = [
            {str(j): data_ij for j, data_ij in enumerate(data_i)}
            for data_i in raw_data_lst
        ]
    del raw_data_lst

    if comment_start is not None:
        data_lst = [
            line
            for line in data_lst
            if not next(iter(line.values())).startswith(comment_start)
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
        msg = f"Invalid argument {orient=}. (expected one of {get_args(Orient)})"
        raise ValueError(msg)

    return result  # type: ignore


def _stringify(x: Any) -> Any:
    if isinstance(x, str):
        return x
    elif isinstance(x, dict):
        return {_stringify(k): _stringify(v) for k, v in x.items()}  # type: ignore
    elif isinstance(x, (list, tuple, set, frozenset)):
        return type(x)(_stringify(xi) for xi in x)
    else:
        return str(x)
