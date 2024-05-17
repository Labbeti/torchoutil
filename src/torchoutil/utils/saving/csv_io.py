#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

from torchoutil.utils.collections import dict_list_to_list_dict
from torchoutil.utils.saving.common import to_builtin


def to_csv(
    data: Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Any]]],
    fpath: Union[str, Path, None],
    *,
    overwrite: bool = True,
    to_builtins: bool = False,
    make_parents: bool = True,
    **csv_writer_kwargs,
) -> str:
    if isinstance(data, Mapping):
        data = dict_list_to_list_dict(data)
    else:
        data = [dict(data_i.items()) for data_i in data]

    if len(data) <= 0:
        raise ValueError(f"Invalid argument {data=}. (found empty iterable)")

    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            fpath.parent.mkdir(parents=True, exist_ok=True)

    if to_builtins:
        data = to_builtin(data)

    fieldnames = list(data[0].keys())  # type: ignore
    file = io.StringIO()
    writer = csv.DictWriter(file, fieldnames, **csv_writer_kwargs)
    writer.writeheader()
    writer.writerows(data)  # type: ignore
    content = file.getvalue()
    file.close()

    if fpath is not None:
        fpath.write_text(content)
    return content


def load_csv(fpath: Union[str, Path], **csv_reader_kwargs) -> List[Dict[str, Any]]:
    with open(fpath, "r") as file:
        reader = csv.DictReader(file, **csv_reader_kwargs)
        data = list(reader)
    return data
