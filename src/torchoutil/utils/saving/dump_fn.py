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
from torchoutil.utils.saving.csv import to_csv
from torchoutil.utils.saving.json import to_json
from torchoutil.utils.saving.pickle import to_pickle

T = TypeVar("T", covariant=True)

DumpFn = Callable[[T, Path], Any]
DumpFnLike = Union[DumpFn[T], SavingBackends]


DUMP_FNS: Dict[str, DumpFn[Any]] = {
    "csv": to_csv,  # type: ignore
    "json": to_json,
    "pickle": to_pickle,
    "torch": torch.save,
}

if _NUMPY_AVAILABLE:
    from .numpy import dump_numpy

    DUMP_FNS["numpy"] = dump_numpy

if _SAFETENSORS_AVAILABLE:
    from .safetensors import to_safetensors

    DUMP_FNS["safetensors"] = to_safetensors

if _YAML_AVAILABLE:
    from .yaml import to_yaml

    DUMP_FNS["yaml"] = to_yaml  # type: ignore


def dump(
    obj: Any,
    fpath: Union[str, Path],
    *args,
    backend: Optional[SavingBackends] = None,
    **kwargs,
) -> None:
    """Load from file using the correct backend."""
    if isinstance(fpath, str):
        fpath = Path(fpath)

    if backend is None:
        ext = fpath.suffix[1:]
        if ext not in EXTENSION_TO_BACKEND.keys():
            msg = f"Unknown extension file {ext}. (expected one of {tuple(EXTENSION_TO_BACKEND.keys())})"
            raise ValueError(msg)
        backend = EXTENSION_TO_BACKEND[ext]

    dump_fn = DUMP_FNS[backend]
    result = dump_fn(obj, fpath, *args, **kwargs)
    return result
