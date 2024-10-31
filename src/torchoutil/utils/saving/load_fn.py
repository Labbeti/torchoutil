#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Callable, Dict, Literal, TypeVar, Union

import torch

from torchoutil.core.packaging import _NUMPY_AVAILABLE, _YAML_AVAILABLE
from torchoutil.types._typing import np
from torchoutil.utils.saving.csv import load_csv
from torchoutil.utils.saving.json import load_json
from torchoutil.utils.saving.pickle import load_pickle
from torchoutil.utils.saving.yaml import load_yaml

T = TypeVar("T", covariant=True)


LoadFn = Callable[[Path], T]
LoadFnLike = Union[
    LoadFn[T], Literal["csv", "json", "numpy", "pickle", "torch", "yaml"]
]


LOAD_FNS: Dict[str, LoadFn[Any]] = {
    "csv": load_csv,
    "json": load_json,
    "pickle": load_pickle,
    "torch": torch.load,
}

if _NUMPY_AVAILABLE:
    LOAD_FNS["numpy"] = np.load

if _YAML_AVAILABLE:
    LOAD_FNS["yaml"] = load_yaml
