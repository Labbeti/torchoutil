#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
from typing import Any


def load_pickle(fpath: Path) -> Any:
    content = pickle.loads(fpath.read_bytes())
    return content


def to_pickle(obj: Any, fpath: Path) -> bytes:
    content = pickle.dumps(obj)
    fpath.write_bytes(content)
    return content
