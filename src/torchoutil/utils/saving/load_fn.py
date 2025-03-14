#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from torchoutil.core.packaging import (
    _NUMPY_AVAILABLE,
    _SAFETENSORS_AVAILABLE,
    _TORCHAUDIO_AVAILABLE,
    _YAML_AVAILABLE,
)
from torchoutil.utils.saving.common import EXTENSION_TO_BACKEND, SavingBackend
from torchoutil.utils.saving.csv import load_csv
from torchoutil.utils.saving.json import load_json
from torchoutil.utils.saving.pickle import load_pickle
from torchoutil.utils.saving.torch import load_torch

T = TypeVar("T", covariant=True)
pylog = logging.getLogger(__name__)

LoadFn = Callable[[Path], T]
LoadFnLike = Union[LoadFn[T], SavingBackend]


LOAD_FNS: Dict[SavingBackend, LoadFn[Any]] = {
    "csv": load_csv,
    "json": load_json,
    "pickle": load_pickle,
    "torch": load_torch,
}

if _NUMPY_AVAILABLE:
    from .numpy import load_numpy

    LOAD_FNS["numpy"] = load_numpy


if _SAFETENSORS_AVAILABLE:
    from ...extras.safetensors import load_safetensors

    LOAD_FNS["safetensors"] = load_safetensors


if _TORCHAUDIO_AVAILABLE:
    from .torchaudio import load_torchaudio

    LOAD_FNS["torchaudio"] = load_torchaudio


if _YAML_AVAILABLE:
    from .yaml import load_yaml

    LOAD_FNS["yaml"] = load_yaml


def load(
    fpath: Union[str, Path, os.PathLike],
    *args,
    saving_backend: Optional[SavingBackend] = None,
    **kwargs,
) -> Any:
    """Load from file using the correct backend."""
    fpath = Path(fpath)

    if not fpath.is_file():
        msg = f"Invalid argument {fpath=}. (path is not a file)"
        raise FileNotFoundError(msg)

    if saving_backend is None:
        ext = fpath.suffix[1:]
        if ext not in EXTENSION_TO_BACKEND.keys():
            msg = f"Unknown extension file '{ext}'. (expected one of {tuple(EXTENSION_TO_BACKEND.keys())} or specify the backend argument with `to.load(..., backend=\"backend\")`)"
            raise ValueError(msg)
        saving_backend = EXTENSION_TO_BACKEND[ext]
        pylog.debug(f"Loading file '{str(fpath)}' using {saving_backend=}.")

    elif saving_backend not in LOAD_FNS:
        msg = f"Invalid argument {saving_backend=}. (expected one of {tuple(LOAD_FNS.keys())})"
        raise ValueError(msg)

    load_fn = LOAD_FNS[saving_backend]
    result = load_fn(fpath, *args, **kwargs)
    return result
