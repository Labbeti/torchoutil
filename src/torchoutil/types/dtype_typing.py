#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import auto
from typing import Dict, Final

import torch
from torch.types import _bool

from pyoutil.enum import StrEnum

TORCH_DTYPES: Final[Dict[str, torch.dtype]] = {
    "float32": torch.float32,
    "float": torch.float,
    "float64": torch.float64,
    "double": torch.double,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "half": torch.half,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.short,
    "int32": torch.int32,
    "int": torch.int,
    "int64": torch.int64,
    "long": torch.long,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "cfloat": torch.cfloat,
    "complex128": torch.complex128,
    "cdouble": torch.cdouble,
    "quint8": torch.quint8,
    "qint8": torch.qint8,
    "qint32": torch.qint32,
    "bool": torch.bool,
    "quint4x2": torch.quint4x2,
}

if hasattr(torch, "chalf"):
    TORCH_DTYPES["chalf"] = torch.chalf
if hasattr(torch, "quint2x4"):
    TORCH_DTYPES["quint2x4"] = torch.quint2x4


class DTypeEnum(StrEnum):
    """Enum of torch dtypes."""

    float16 = auto()
    float32 = auto()
    float64 = auto()
    int16 = auto()
    int32 = auto()
    int64 = auto()
    complex32 = auto()
    complex64 = auto()
    complex128 = auto()

    float = float32
    double = float64
    half = float16
    short = int16
    int = int32
    long = int64
    chalf = complex32
    cfloat = complex64
    cdouble = complex128

    bfloat16 = auto()
    uint8 = auto()  # byte
    int8 = auto()  # char
    quint8 = auto()
    qint8 = auto()
    qint32 = auto()
    bool = auto()
    quint4x2 = auto()
    quint2x4 = auto()

    @classmethod
    def default(cls) -> "DTypeEnum":
        return cls.from_dtype(torch.get_default_dtype())

    @classmethod
    def from_dtype(cls, dtype: torch.dtype) -> "DTypeEnum":
        for name_i, dtype_i in TORCH_DTYPES.items():
            if dtype_i == dtype:
                return DTypeEnum.from_str(name_i)

        raise ValueError(
            f"Invalid argument {dtype=}. (expected one of {tuple(TORCH_DTYPES.keys())})"
        )

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


def torch_dtype_to_enum_dtype(dtype: torch.dtype) -> DTypeEnum:
    return DTypeEnum.from_dtype(dtype)


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype)


def str_to_enum_dtype(dtype: str) -> DTypeEnum:
    return DTypeEnum.from_str(dtype)


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    return TORCH_DTYPES[dtype]


def enum_dtype_to_str(dtype: DTypeEnum) -> str:
    return str(dtype)


def enum_dtype_to_torch_dtype(dtype: DTypeEnum) -> torch.dtype:
    return dtype.dtype
