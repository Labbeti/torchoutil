#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, overload, Literal

from torchoutil.core.packaging import (
    _NUMPY_AVAILABLE,
    _SAFETENSORS_AVAILABLE,
    _TORCHAUDIO_AVAILABLE,
    _YAML_AVAILABLE,
)
from torchoutil.utils.saving.common import EXTENSION_TO_BACKEND, SavingBackend
from torchoutil.utils.saving.csv import dump_csv
from torchoutil.utils.saving.json import dump_json
from torchoutil.utils.saving.pickle import dump_pickle
from torchoutil.utils.saving.torch import to_torch


DumpFn = Callable[..., Union[str, bytes]]
DumpFnLike = Union[DumpFn, SavingBackend]


DUMP_FNS: Dict[str, DumpFn] = {
    "csv": dump_csv,  # type: ignore
    "json": dump_json,
    "pickle": dump_pickle,
    "torch": to_torch,
}

if _NUMPY_AVAILABLE:
    from .numpy import dump_numpy

    DUMP_FNS["numpy"] = dump_numpy

if _SAFETENSORS_AVAILABLE:
    from ...extras.safetensors import dump_safetensors

    DUMP_FNS["safetensors"] = dump_safetensors

if _TORCHAUDIO_AVAILABLE:
    from .torchaudio import dump_torchaudio

    DUMP_FNS["torchaudio"] = dump_torchaudio

if _YAML_AVAILABLE:
    from .yaml import dump_yaml

    DUMP_FNS["yaml"] = dump_yaml


@overload
def dump(
    obj: Any,
    fpath: Union[str, Path, os.PathLike],
    *args,
    saving_backend: Optional[SavingBackend] = None,
    **kwargs,
) -> Union[str, bytes]: ...


@overload
def dump(
    obj: Any,
    fpath: Literal[None] = None,
    *args,
    saving_backend: SavingBackend,
    **kwargs,
) -> Union[str, bytes]: ...


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
        if fpath is None:
            msg = f"Invalid combinaison of arguments {fpath=} and {saving_backend=}."
            raise ValueError(msg)

        ext = fpath.suffix[1:]
        if ext not in EXTENSION_TO_BACKEND.keys():
            msg = f"Unknown extension file {ext}. (expected one of {tuple(EXTENSION_TO_BACKEND.keys())})"
            raise ValueError(msg)
        saving_backend = EXTENSION_TO_BACKEND[ext]

    dump_fn = DUMP_FNS[saving_backend]
    result = dump_fn(obj, fpath, *args, **kwargs)
    return result


# Alias for dump
save = dump
