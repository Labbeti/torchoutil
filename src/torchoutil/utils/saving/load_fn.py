#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Literal, TypeVar, Union

import torch

from torchoutil.core.packaging import _NUMPY_AVAILABLE, _YAML_AVAILABLE
from torchoutil.types._typing import np
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
    "pickle": pickle_load_fn,
    "torch": torch.load,
}

if _NUMPY_AVAILABLE:
    LOAD_FNS["numpy"] = np.load

if _YAML_AVAILABLE:
    LOAD_FNS["yaml"] = load_yaml
