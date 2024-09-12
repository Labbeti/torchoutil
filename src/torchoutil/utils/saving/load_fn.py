#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Literal, TypeVar, Union

import torch

from torchoutil.types import np
from torchoutil.utils.saving.csv_io import load_csv
from torchoutil.utils.saving.yaml_io import load_yaml

T = TypeVar("T", covariant=True)


LoadFn = Callable[[Path], T]
LoadFnLike = Union[
    LoadFn[T], Literal["csv", "json", "numpy", "pickle", "torch", "yaml"]
]


def json_load_fn(fpath: Path) -> Any:
    return json.loads(fpath.read_text())


def pickle_load_fn(fpath: Path) -> Any:
    return pickle.loads(fpath.read_bytes())


LOAD_FNS: Dict[str, LoadFn[Any]] = {
    "csv": load_csv,
    "json": json_load_fn,
    "numpy": np.load,
    "pickle": pickle_load_fn,
    "torch": torch.load,
    "yaml": load_yaml,
}
