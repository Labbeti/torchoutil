#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import Namespace
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import Any, Dict, Iterable, List, Literal, Mapping, TypeVar, Union, overload

import torch
from torch import Tensor

from torchoutil.core.packaging import (
    _NUMPY_AVAILABLE,
    _OMEGACONF_AVAILABLE,
    _PANDAS_AVAILABLE,
)
from torchoutil.pyoutil.typing import (
    BuiltinNumber,
    DataclassInstance,
    NamedTupleInstance,
    T_BuiltinScalar,
    is_builtin_scalar,
    is_dataclass_instance,
    is_namedtuple_instance,
)

if _NUMPY_AVAILABLE:
    import numpy as np

if _OMEGACONF_AVAILABLE:
    from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore

if _PANDAS_AVAILABLE:
    from pandas import DataFrame


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

UnkMode = Literal["identity", "error"]
UNK_MODES = ("identity", "error")


@overload
def to_builtin(
    x: T_BuiltinScalar,
    *,
    unk_mode: UnkMode = "error",
) -> T_BuiltinScalar:
    ...


@overload
def to_builtin(
    x: Enum,
    *,
    unk_mode: UnkMode = "error",
) -> str:
    ...


@overload
def to_builtin(
    x: Path,
    *,
    unk_mode: UnkMode = "error",
) -> str:
    ...


@overload
def to_builtin(
    x: Pattern,
    *,
    unk_mode: UnkMode = "error",
) -> str:
    ...


@overload
def to_builtin(
    x: Namespace,
    *,
    unk_mode: UnkMode = "error",
) -> Dict[str, Any]:
    ...


@overload
def to_builtin(
    x: Tensor,
    *,
    unk_mode: UnkMode = "error",
) -> Union[List, BuiltinNumber]:
    ...


@overload
def to_builtin(
    x: Mapping[K, V],
    *,
    unk_mode: UnkMode = "error",
) -> Dict[K, V]:
    ...


@overload
def to_builtin(
    x: DataclassInstance,
    *,
    unk_mode: UnkMode = "error",
) -> Dict[str, Any]:
    ...


@overload
def to_builtin(
    x: NamedTupleInstance,
    *,
    unk_mode: UnkMode = "error",
) -> Dict[str, Any]:
    ...


@overload
def to_builtin(
    x: Any,
    *,
    unk_mode: UnkMode = "error",
) -> Any:
    ...


def to_builtin(
    x: Any,
    *,
    unk_mode: UnkMode = "error",
) -> Any:
    """Helper function to sanitize data before saving to YAML or CSV file.

    Args:
        x: Object to convert to built-in equivalent.
        unk_mode: When an object type is not recognized, unk_mode defines the behaviour.
            If unk_mode == "identity", the object is returned unchanged.
            If unk_mode == "error", a TypeError is raised.
    """
    # Terminal cases
    if is_builtin_scalar(x, strict=True):
        return x
    if isinstance(x, Enum):
        return x.name
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Pattern):
        return x.pattern
    if isinstance(x, Tensor):
        return x.tolist()
    if isinstance(x, torch.dtype):
        return str(x)
    if _NUMPY_AVAILABLE and isinstance(x, np.ndarray):  # type: ignore
        return x.tolist()
    if _NUMPY_AVAILABLE and isinstance(x, np.generic):  # type: ignore
        return x.item()
    if _NUMPY_AVAILABLE and isinstance(x, np.dtype):  # type: ignore
        return str(x)

    # Non-terminal cases (iterables and mappings)
    if _OMEGACONF_AVAILABLE and isinstance(x, (DictConfig, ListConfig)):  # type: ignore
        return to_builtin(OmegaConf.to_container(x, resolve=False, enum_to_str=True))  # type: ignore
    if _PANDAS_AVAILABLE and isinstance(x, DataFrame):  # type: ignore
        return to_builtin(x.to_dict("list"))
    if isinstance(x, Namespace):
        return to_builtin(x.__dict__)
    if is_dataclass_instance(x):
        return to_builtin(asdict(x))
    if is_namedtuple_instance(x):
        return to_builtin(x._asdict())
    if isinstance(x, Mapping):
        return {to_builtin(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, Iterable):
        return [to_builtin(xi) for xi in x]

    if unk_mode == "identity":
        return x
    elif unk_mode == "error":
        raise TypeError(f"Unsupported argument type {type(x)}.")
    else:
        raise ValueError(f"Invalid argument {unk_mode=}. (expected one of {UNK_MODES})")
