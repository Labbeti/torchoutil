#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import torch

from torchoutil.core.packaging import (
    _NUMPY_AVAILABLE,
    _SAFETENSORS_AVAILABLE,
    _YAML_AVAILABLE,
)
from torchoutil.utils.saving.common import EXTENSION_TO_BACKEND, SavingBackends
from torchoutil.utils.saving.csv import load_csv
from torchoutil.utils.saving.json import load_json
from torchoutil.utils.saving.pickle import load_pickle

T = TypeVar("T", covariant=True)


LoadFn = Callable[[Path], T]
LoadFnLike = Union[LoadFn[T], SavingBackends]


LOAD_FNS: Dict[SavingBackends, LoadFn[Any]] = {
    "csv": load_csv,
    "json": load_json,
    "pickle": load_pickle,
    "torch": torch.load,
}

if _NUMPY_AVAILABLE:
    from .numpy import load_numpy

    LOAD_FNS["numpy"] = load_numpy


if _SAFETENSORS_AVAILABLE:
    from .safetensors import load_safetensors

    LOAD_FNS["safetensors"] = load_safetensors


if _YAML_AVAILABLE:
    from .yaml import load_yaml

    LOAD_FNS["yaml"] = load_yaml


def load(
    fpath: Union[str, Path],
    *args,
    backend: Optional[SavingBackends] = None,
    **kwargs,
) -> Any:
    """Load from file using the correct backend."""
    fpath = Path(fpath)

    if backend is None:
        ext = fpath.suffix[1:]
        if ext not in EXTENSION_TO_BACKEND.keys():
            msg = f"Unknown extension file {ext}. (expected one of {tuple(EXTENSION_TO_BACKEND.keys())})"
            raise ValueError(msg)
        backend = EXTENSION_TO_BACKEND[ext]

    load_fn = LOAD_FNS[backend]
    result = load_fn(fpath, *args, **kwargs)
    return result
