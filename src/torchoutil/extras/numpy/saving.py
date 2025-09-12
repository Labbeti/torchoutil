#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union, get_args

from torchoutil.pyoutil.io import _setup_path

from .definitions import np

NumpyFormat = Literal["npy", "npz"]


def dump_numpy(
    obj: np.ndarray,
    fpath: Union[str, Path, None] = None,
    *args,
    np_format: Optional[NumpyFormat] = "npy",
    overwrite: bool = True,
    make_parents: bool = True,
    **kwargs,
) -> bytes:
    fpath = _setup_path(fpath, overwrite, make_parents)

    if np_format is not None:
        pass
    elif fpath is None or fpath.suffix == ".npy":
        np_format = "npy"
    elif fpath.suffix == ".npz":
        np_format = "npz"
    else:
        msg = f"Unknown numpy extension '{fpath.suffix}'. (expected one of {get_args(NumpyFormat)})"
        warnings.warn(msg)
        np_format = "npy"

    if np_format == "npy":
        save_fn = np.save
    elif np_format == "npz":
        save_fn = np.savez
    else:
        raise ValueError(f"Invalid argument {np_format=}.")

    buffer = BytesIO()
    save_fn(buffer, obj, *args, **kwargs)
    buffer.seek(0)
    content = buffer.read()

    if fpath is not None:
        fpath.write_bytes(content)
    return content


def load_numpy(fpath: Union[str, Path], *args, **kwargs) -> np.ndarray:
    return np.load(fpath, *args, **kwargs)
