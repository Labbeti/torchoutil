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
from torchoutil.utils.saving.csv import load_csv
from torchoutil.utils.saving.json import load_json
from torchoutil.utils.saving.pickle import load_pickle

T = TypeVar("T", covariant=True)


LoadFn = Callable[[Path], T]
LoadFnName = Literal["csv", "json", "numpy", "pickle", "safetensors", "torch", "yaml"]
LoadFnLike = Union[LoadFn[T], LoadFnName]


LOAD_FNS: Dict[LoadFnName, LoadFn[Any]] = {
    "csv": load_csv,
    "json": load_json,
    "pickle": load_pickle,
    "torch": torch.load,
}

if _NUMPY_AVAILABLE:
    LOAD_FNS["numpy"] = np.load


if _SAFETENSORS_AVAILABLE:
    from torchoutil.utils.saving.safetensors import load_safetensors

    LOAD_FNS["safetensors"] = load_safetensors


if _YAML_AVAILABLE:
    from torchoutil.utils.saving.yaml import load_yaml

    LOAD_FNS["yaml"] = load_yaml
