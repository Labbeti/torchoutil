#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from typing_extensions import TypeAlias

from torchoutil.core.packaging import (
    _NUMPY_AVAILABLE,
    _SAFETENSORS_AVAILABLE,
    _TENSORBOARD_AVAILABLE,
    _TORCHAUDIO_AVAILABLE,
    _YAML_AVAILABLE,
)

from .common import SavingBackend, _fpath_to_saving_backend
from .csv import load_csv
from .json import load_json
from .pickle import load_pickle
from .torch import load_torch

T = TypeVar("T", covariant=True)
pylog = logging.getLogger(__name__)

LoadFn: TypeAlias = Callable[[Path], T]
LoadFnLike: TypeAlias = Union[LoadFn[T], SavingBackend]


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
    from torchoutil.extras.safetensors import load_safetensors

    LOAD_FNS["safetensors"] = load_safetensors


if _TENSORBOARD_AVAILABLE:
    from torchoutil.extras.tensorboard import load_tfevents

    LOAD_FNS["tensorboard"] = load_tfevents


if _TORCHAUDIO_AVAILABLE:
    from .torchaudio import load_with_torchaudio

    LOAD_FNS["torchaudio"] = load_with_torchaudio


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
        saving_backend = _fpath_to_saving_backend(fpath)

    elif saving_backend not in LOAD_FNS:
        msg = f"Invalid argument {saving_backend=}. (expected one of {tuple(LOAD_FNS.keys())})"
        raise ValueError(msg)

    load_fn = LOAD_FNS[saving_backend]
    result = load_fn(fpath, *args, **kwargs)
    return result
