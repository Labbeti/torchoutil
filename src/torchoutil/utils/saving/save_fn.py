#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Callable, Dict, Literal, TypeVar, Union

import torch

from torchoutil.core.packaging import _NUMPY_AVAILABLE, _YAML_AVAILABLE
from torchoutil.types._typing import np
from torchoutil.utils.saving.csv import to_csv
from torchoutil.utils.saving.json import to_json
from torchoutil.utils.saving.pickle import to_pickle

T = TypeVar("T", covariant=True)

SaveFn = Callable[[T, Path], None]
SaveFnLike = Union[
    SaveFn[T], Literal["csv", "json", "numpy", "pickle", "torch", "yaml"]
]


def numpy_save_fn(obj: Any, fpath: Path) -> None:
    np.save(fpath, obj)


SAVE_FNS: Dict[str, SaveFn[Any]] = {
    "csv": to_csv,  # type: ignore
    "json": to_json,
    "pickle": to_pickle,
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
    from torchoutil.utils.saving.yaml import to_yaml

    SAVE_FNS["yaml"] = to_yaml  # type: ignore
    SAVE_EXTENSIONS["yaml"] = "yaml"
