#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
from typing import Any, Union


def load_pickle(fpath: Union[str, Path]) -> Any:
    fpath = Path(fpath)
    content = pickle.loads(fpath.read_bytes())
    return content


def to_pickle(obj: Any, fpath: Union[str, Path]) -> bytes:
    fpath = Path(fpath)
    content = pickle.dumps(obj)
    fpath.write_bytes(content)
    return content
