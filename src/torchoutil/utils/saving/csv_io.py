#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

from torchoutil.utils.collections import dict_list_to_list_dict
from torchoutil.utils.saving.common import to_builtin


def save_to_csv(
    data: Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Any]]],
    fpath: Union[str, Path, None],
    *,
    overwrite: bool = True,
    to_builtins: bool = True,
    make_parents: bool = True,
    **kwargs,
) -> str:
    if isinstance(data, Mapping):
        data = dict_list_to_list_dict(data)
    else:
        data = list(data)

    if len(data) <= 0:
        raise ValueError(f"Invalid argument {data=}. (found empty iterable)")

    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            os.makedirs(fpath.parent, exist_ok=True)

    if to_builtins:
        data = to_builtin(data)

    fieldnames = list(data[0].keys())  # type: ignore
    file = io.StringIO()
    writer = csv.DictWriter(file, fieldnames, **kwargs)
    writer.writeheader()
    writer.writerows(data)  # type: ignore
    file.close()
    content = file.getvalue()

    if fpath is not None:
        fpath.write_text(content)
    return content


def load_csv(fpath: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
    with open(fpath, "r") as file:
        reader = csv.DictReader(file, **kwargs)
        data = list(reader)
    return data
