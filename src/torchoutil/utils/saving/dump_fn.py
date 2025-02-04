#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from torchoutil.core.packaging import (
    _NUMPY_AVAILABLE,
    _SAFETENSORS_AVAILABLE,
    _YAML_AVAILABLE,
    _TORCHAUDIO_AVAILABLE,
)
from torchoutil.utils.saving.common import EXTENSION_TO_BACKEND, SavingBackend
from torchoutil.utils.saving.csv import to_csv
from torchoutil.utils.saving.json import to_json
from torchoutil.utils.saving.pickle import to_pickle
from torchoutil.utils.saving.torch import to_torch


T = TypeVar("T", covariant=True)

DumpFn = Callable[[T, Path], Union[str, bytes]]
DumpFnLike = Union[DumpFn[T], SavingBackend]


DUMP_FNS: Dict[str, DumpFn[Any]] = {
    "csv": to_csv,  # type: ignore
    "json": to_json,
    "pickle": to_pickle,
    "torch": to_torch,
}

if _NUMPY_AVAILABLE:
    from .numpy import dump_numpy

    DUMP_FNS["numpy"] = dump_numpy

if _SAFETENSORS_AVAILABLE:
    from .safetensors import to_safetensors

    DUMP_FNS["safetensors"] = to_safetensors

if _TORCHAUDIO_AVAILABLE:
    from .torchaudio import to_torchaudio

    DUMP_FNS["torchaudio"] = to_torchaudio

if _YAML_AVAILABLE:
    from .yaml import to_yaml

    DUMP_FNS["yaml"] = to_yaml


def dump(
    obj: Any,
    fpath: Union[str, Path, os.PathLike, None] = None,
    *args,
    saving_backend: Optional[SavingBackend] = None,
    **kwargs,
) -> Union[str, bytes]:
    """Load from file using the correct backend."""
    if isinstance(fpath, (str, os.PathLike)):
        fpath = Path(fpath)

    if saving_backend is None:
        ext = fpath.suffix[1:]
        if ext not in EXTENSION_TO_BACKEND.keys():
            msg = f"Unknown extension file {ext}. (expected one of {tuple(EXTENSION_TO_BACKEND.keys())})"
            raise ValueError(msg)
        saving_backend = EXTENSION_TO_BACKEND[ext]

    dump_fn = DUMP_FNS[saving_backend]
    result = dump_fn(obj, fpath, *args, **kwargs)
    return result


save = dump
