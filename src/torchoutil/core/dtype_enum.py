#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import auto
from typing import Any, Dict, Final

import torch
from torch.types import _bool, _int

from torchoutil.pyoutil.enum import StrEnum

TORCH_DTYPES: Final[Dict[str, torch.dtype]] = {
    # Base types
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    # Aliases
    "half": torch.half,
    "float": torch.float,
    "double": torch.double,
    "short": torch.short,
    "int": torch.int,
    "long": torch.long,
    "chalf": torch.chalf if hasattr(torch, "chalf") else torch.complex32,
    "cfloat": torch.cfloat,
    "cdouble": torch.cdouble,
    # Others
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "int8": torch.int8,
    "qint8": torch.qint8,
    "qint32": torch.qint32,
    "quint4x2": torch.quint4x2,
    "quint8": torch.quint8,
    "uint8": torch.uint8,
}

# Optional
if hasattr(torch, "quint2x4"):
    TORCH_DTYPES["quint2x4"] = torch.quint2x4
if hasattr(torch, "uint16"):
    TORCH_DTYPES["uint16"] = torch.uint16
if hasattr(torch, "uint32"):
    TORCH_DTYPES["uint32"] = torch.uint32
if hasattr(torch, "uint64"):
    TORCH_DTYPES["uint64"] = torch.uint64


_NAME_TO_DTYPE: Final[Dict[str, torch.dtype]] = TORCH_DTYPES
_DTYPE_TO_NAME: Final[Dict[torch.dtype, str]] = {
    dt: name for name, dt in _NAME_TO_DTYPE.items()
}


class DTypeEnum(StrEnum):
    """Enum of torch dtypes."""

    # Base types
    float16 = auto()
    float32 = auto()
    float64 = auto()
    int16 = auto()
    int32 = auto()
    int64 = auto()
    complex32 = auto()
    complex64 = auto()
    complex128 = auto()

    # Aliases
    half = float16
    float = float32
    double = float64
    short = int16
    int = int32
    long = int64
    chalf = complex32
    cfloat = complex64
    cdouble = complex128

    # Others
    bfloat16 = auto()
    bool = auto()
    int8 = auto()  # char
    qint8 = auto()
    qint32 = auto()
    quint4x2 = auto()
    quint8 = auto()
    uint8 = auto()  # byte

    # Optional
    quint2x4 = auto()
    uint16 = auto()
    uint32 = auto()
    uint64 = auto()

    @classmethod
    def default(cls) -> "DTypeEnum":
        return cls.from_dtype(torch.get_default_dtype())

    @classmethod
    def from_dtype(cls, dtype: torch.dtype) -> "DTypeEnum":
        if dtype not in _DTYPE_TO_NAME:
            msg = f"Invalid argument {dtype=}. (expected one of {tuple(_DTYPE_TO_NAME.keys())})"
            raise ValueError(msg)
        return DTypeEnum.from_str(_DTYPE_TO_NAME[dtype])

    @property
    def dtype(self) -> torch.dtype:
        return TORCH_DTYPES[self.name]

    @property
    def is_floating_point(self) -> _bool:
        return self.dtype.is_floating_point

    @property
    def is_complex(self) -> _bool:
        return self.dtype.is_complex

    @property
    def is_signed(self) -> _bool:
        return self.dtype.is_signed

    @property
    def itemsize(self) -> _int:
        return self.dtype.itemsize

    def to_real(self) -> "DTypeEnum":
        return DTypeEnum.from_dtype(self.dtype.to_real())

    def to_complex(self) -> "DTypeEnum":
        return DTypeEnum.from_dtype(self.dtype.to_complex())

    def __eq__(self, other: Any) -> _bool:
        if isinstance(other, DTypeEnum):
            return self.dtype == other.dtype
        elif isinstance(other, torch.dtype):
            return self.dtype == other
        elif isinstance(other, str):
            return self.dtype == str_to_torch_dtype(other)
        else:
            return False

    def __hash__(self) -> _int:
        return hash(self.dtype)


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    return _removeprefix(str(dtype), "torch.")


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    return _NAME_TO_DTYPE[_removeprefix(dtype, "torch.")]


def torch_dtype_to_enum_dtype(dtype: torch.dtype) -> DTypeEnum:
    return DTypeEnum.from_dtype(dtype)


def str_to_enum_dtype(dtype: str) -> DTypeEnum:
    return DTypeEnum.from_str(dtype)


def enum_dtype_to_str(dtype: DTypeEnum) -> str:
    return str(dtype)


def enum_dtype_to_torch_dtype(dtype: DTypeEnum) -> torch.dtype:
    return dtype.dtype


def _removeprefix(x: str, prefix: str) -> str:
    # str.removeprefix does not exists in 3.8, so we use this function instead
    if x.startswith(prefix):
        return x[len(prefix) :]
    else:
        return x
