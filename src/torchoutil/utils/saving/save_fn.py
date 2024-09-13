#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Literal, TypeVar, Union

import torch

from torchoutil.types._typing import np
from torchoutil.utils.saving.csv_io import to_csv
from torchoutil.utils.saving.yaml_io import to_yaml

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


SAVE_FNS: Dict[str, SaveFn[Any]] = {  # type: ignore
    "csv": to_csv,
    "json": json_save_fn,
    "numpy": numpy_save_fn,
    "pickle": pickle_save_fn,
    "torch": torch.save,
    "yaml": to_yaml,
}
SAVE_EXTENSIONS = {
    "csv": "csv",
    "json": "json",
    "numpy": "npy",
    "pickle": "pickle",
    "torch": "pt",
    "yaml": "yaml",
}
