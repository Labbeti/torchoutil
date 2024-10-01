#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Literal, TypeVar, Union

import torch

from torchoutil.core.packaging import _NUMPY_AVAILABLE, _YAML_AVAILABLE
from torchoutil.types._typing import np
from torchoutil.utils.saving.csv_io import to_csv

T = TypeVar("T", covariant=True)

SaveFn = Callable[[T, Path], None]
SaveFnLike = Union[
    SaveFn[T], Literal["csv", "json", "numpy", "pickle", "torch", "yaml"]
]


def json_save_fn(obj: Any, fpath: Path) -> None:
    fpath.write_text(json.dumps(obj))


def numpy_save_fn(obj: Any, fpath: Path) -> None:
    np.save(fpath, obj)


def pickle_save_fn(obj: Any, fpath: Path) -> None:
    fpath.write_bytes(pickle.dumps(obj))


SAVE_FNS: Dict[str, SaveFn[Any]] = {
    "csv": to_csv,  # type: ignore
    "json": json_save_fn,
    "pickle": pickle_save_fn,
    "torch": torch.save,
}
SAVE_EXTENSIONS = {
    "csv": "csv",
    "json": "json",
    "pickle": "pickle",
    "torch": "pt",
}

if _NUMPY_AVAILABLE:
    SAVE_FNS["numpy"] = numpy_save_fn
    SAVE_EXTENSIONS["numpy"] = "npy"


if _YAML_AVAILABLE:
    from torchoutil.utils.saving.yaml_io import to_yaml

    SAVE_FNS["yaml"] = to_yaml  # type: ignore
    SAVE_EXTENSIONS["yaml"] = "yaml"
