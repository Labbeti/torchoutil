#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BytesIO
from pathlib import Path
from typing import Union

from torchoutil.extras.numpy import np


def dump_numpy(
    obj: np.ndarray, fpath: Union[str, Path, None] = None, *args, **kwargs
) -> bytes:
    buffer = BytesIO()
    np.save(buffer, obj, *args, **kwargs)
    buffer.seek(0)
    content = buffer.read()

    if fpath is not None:
        fpath = Path(fpath)
        fpath.write_bytes(content)
    return content


def load_numpy(fpath: Union[str, Path], *args, **kwargs) -> np.ndarray:
    return np.load(fpath, *args, **kwargs)
