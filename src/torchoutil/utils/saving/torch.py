#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import torch
from torch.serialization import FILE_LIKE, MAP_LOCATION, DEFAULT_PROTOCOL

from torchoutil.pyoutil.io import _setup_path
from torchoutil.pyoutil.typing import NoneType


def to_torch(
    obj: object,
    f: Optional[FILE_LIKE] = None,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
) -> bytes:
    if isinstance(f, (str, Path, os.PathLike, NoneType)):
        f = _setup_path(f, overwrite, make_parents)
        buffer = io.BytesIO()
    else:
        buffer = f

    torch.save(
        obj,
        buffer,
        pickle_module,
        pickle_protocol,
        _use_new_zipfile_serialization,
        _disable_byteorder_record,
    )

    content = buffer.getvalue()
    if isinstance(f, Path):
        f.write_bytes(content)

    return content


def load_torch(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = False,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any,
) -> Any:
    return torch.load(
        f,
        map_location,
        pickle_module,
        weights_only=weights_only,
        mmap=mmap,
        **pickle_load_args,
    )
