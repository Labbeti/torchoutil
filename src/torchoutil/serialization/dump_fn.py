#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Optional, Union, overload

from typing_extensions import TypeAlias

from torchoutil.core.packaging import (
    _H5PY_AVAILABLE,
    _NUMPY_AVAILABLE,
    _SAFETENSORS_AVAILABLE,
    _TORCHAUDIO_AVAILABLE,
    _YAML_AVAILABLE,
)
from torchoutil.pyoutil.functools import function_alias

from .common import SavingBackend, _fpath_to_saving_backend
from .csv import dump_csv
from .json import dump_json
from .pickle import dump_pickle
from .torch import dump_torch

DumpFn: TypeAlias = Callable[..., Any]
DumpFnLike: TypeAlias = Union[DumpFn, SavingBackend]


DUMP_FNS: Dict[SavingBackend, DumpFn] = {
    "csv": dump_csv,  # type: ignore
    "json": dump_json,
    "pickle": dump_pickle,
    "torch": dump_torch,
}

if _H5PY_AVAILABLE:
    from .hdf import dump_hdf

    DUMP_FNS["h5py"] = dump_hdf

if _NUMPY_AVAILABLE:
    from .numpy import dump_numpy

    DUMP_FNS["numpy"] = dump_numpy

if _SAFETENSORS_AVAILABLE:
    from torchoutil.extras.safetensors import dump_safetensors

    DUMP_FNS["safetensors"] = dump_safetensors

if _TORCHAUDIO_AVAILABLE:
    from .torchaudio import dump_with_torchaudio

    DUMP_FNS["torchaudio"] = dump_with_torchaudio

if _YAML_AVAILABLE:
    from .yaml import dump_yaml

    DUMP_FNS["yaml"] = dump_yaml


@overload
def dump(
    obj: Any,
    fpath: Union[None, BinaryIO] = None,
    *args,
    saving_backend: SavingBackend = "torch",
    **kwargs,
) -> Union[str, bytes]:
    ...


@overload
def dump(
    obj: Any,
    fpath: Union[str, Path, os.PathLike],
    *args,
    saving_backend: Optional[SavingBackend] = "torch",
    **kwargs,
) -> Union[str, bytes]:
    ...


def dump(
    obj: Any,
    fpath: Union[str, Path, os.PathLike, None, BinaryIO] = None,
    *args,
    saving_backend: Optional[SavingBackend] = "torch",
    **kwargs,
) -> Union[str, bytes]:
    """Load from file using the correct backend."""
    if isinstance(fpath, (str, os.PathLike)):
        fpath = Path(fpath)

    if saving_backend is None:
        if not isinstance(fpath, (str, Path, os.PathLike)):
            msg = f"Invalid combinaison of arguments {fpath=} and {saving_backend=}."
            raise ValueError(msg)

        saving_backend = _fpath_to_saving_backend(fpath)

    elif saving_backend not in DUMP_FNS:
        msg = f"Invalid argument {saving_backend=}. (expected one of {tuple(DUMP_FNS.keys())})"
        raise ValueError(msg)

    dump_fn = DUMP_FNS[saving_backend]
    result = dump_fn(obj, fpath, *args, **kwargs)
    return result


@function_alias(dump)
def save(*args, **kwargs):
    ...
