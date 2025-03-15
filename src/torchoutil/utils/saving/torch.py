#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import io
import os
import pickle
from pathlib import Path
from typing import IO, Any, BinaryIO, Callable, Dict, Optional, Union

import torch
from torch.serialization import DEFAULT_PROTOCOL
from torch.types import Storage
from typing_extensions import TypeAlias

from torchoutil.core.semver import Version
from torchoutil.pyoutil.io import _setup_path

FileLike: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
MapLocationLike: TypeAlias = Optional[
    Union[Callable[[Storage, str], Storage], torch.device, str, Dict[str, str]]
]


def to_torch(
    obj: object,
    f: Optional[FileLike] = None,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
) -> bytes:
    if isinstance(f, (str, Path, os.PathLike)) or f is None:
        f = _setup_path(f, overwrite, make_parents)
        buffer = io.BytesIO()
    else:
        buffer = f

    if "_disable_byteorder_record" in inspect.getargs(torch.save.__code__).args:
        kwds = dict(_disable_byteorder_record=_disable_byteorder_record)
    else:
        kwds = {}

    torch.save(
        obj,
        buffer,
        pickle_module,
        pickle_protocol,
        _use_new_zipfile_serialization,
        **kwds,
    )

    if isinstance(buffer, io.BytesIO):
        content = buffer.getvalue()
    else:
        content = buffer.read()

    if isinstance(f, Path):
        f.write_bytes(content)

    return content


def load_torch(
    f: FileLike,
    map_location: MapLocationLike = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = False,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any,
) -> Any:
    kwds = {}
    if Version(torch.__version__) >= Version("2.0.0"):
        kwds |= dict(
            weights_only=weights_only,
            mmap=mmap,
        )
    else:
        pickle_module = pickle

    return torch.load(
        f,
        map_location,
        pickle_module,
        **kwds,
        **pickle_load_args,
    )
