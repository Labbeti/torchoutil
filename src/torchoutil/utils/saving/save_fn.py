#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Callable, Dict, Literal, TypeVar, Union

import torch

from torchoutil.core.packaging import (
    _NUMPY_AVAILABLE,
    _SAFETENSORS_AVAILABLE,
    _YAML_AVAILABLE,
)
from torchoutil.extras.numpy import np
from torchoutil.utils.saving.csv import to_csv
from torchoutil.utils.saving.json import to_json
from torchoutil.utils.saving.pickle import to_pickle

T = TypeVar("T", covariant=True)

SaveFn = Callable[[T, Path], None]
SafeFnName = Literal["csv", "json", "numpy", "pickle", "safetensors", "torch", "yaml"]
SaveFnLike = Union[SaveFn[T], SafeFnName]


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

if _SAFETENSORS_AVAILABLE:
    from torchoutil.utils.saving.safetensors import to_safetensors

    SAVE_FNS["safetensors"] = to_safetensors
    SAVE_EXTENSIONS["safetensors"] = "safetensors"

if _YAML_AVAILABLE:
    from torchoutil.utils.saving.yaml import to_yaml

    SAVE_FNS["yaml"] = to_yaml  # type: ignore
    SAVE_EXTENSIONS["yaml"] = "yaml"
