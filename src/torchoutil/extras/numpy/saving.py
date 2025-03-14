#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from io import BytesIO
from pathlib import Path
from typing import Union

from torchoutil.extras.numpy import np
from torchoutil.pyoutil.io import _setup_path


def dump_numpy(
    obj: np.ndarray,
    fpath: Union[str, Path, None] = None,
    *args,
    overwrite: bool = True,
    make_parents: bool = True,
    **kwargs,
) -> bytes:
    fpath = _setup_path(fpath, overwrite, make_parents)

    if fpath is None or fpath.suffix == ".npy":
        save_fn = np.save
    elif fpath.suffix == ".npz":
        save_fn = np.savez
    else:
        NUMPY_EXTENSIONS = (".npy", ".npz")
        msg = f"Unknown numpy extension '{fpath.suffix}'. (expected one of {NUMPY_EXTENSIONS})"
        warnings.warn(msg)
        save_fn = np.save

    buffer = BytesIO()
    save_fn(buffer, obj, *args, **kwargs)
    buffer.seek(0)
    content = buffer.read()

    if fpath is not None:
        fpath.write_bytes(content)
    return content


def load_numpy(fpath: Union[str, Path], *args, **kwargs) -> np.ndarray:
    return np.load(fpath, *args, **kwargs)
