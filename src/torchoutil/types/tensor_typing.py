#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tensor subclasses for typing and instance checks.

Note: torchoutil.FloatTensor != torch.FloatTensor but issubclass(torchoutil.FloatTensor, torch.FloatTensor) is False because torch.FloatTensor cannot be subclassed
"""

from enum import auto
from typing import (
    Any,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

import torch
from torch._C import _TensorMeta
from torch.types import Device, _bool

from pyoutil import BuiltinNumber, StrEnum, TBuiltinNumber
from torchoutil.nn import functional as F
from torchoutil.types.classes import TORCH_DTYPES

T_DType = TypeVar("T_DType", "DTypeEnum", None)
T_NDim = TypeVar("T_NDim", bound=int)
T_Tensor = TypeVar("T_Tensor", bound=torch.Tensor)

_DEFAULT_T_DTYPE = Literal[None]
_DEFAULT_T_NDIM = int

_TORCH_BASE_CLASSES: Final[Dict[str, Type]] = {
    "float32": torch.FloatTensor,
    "float": torch.FloatTensor,
    "float64": torch.DoubleTensor,
    "double": torch.DoubleTensor,
    "float16": torch.HalfTensor,
    "half": torch.HalfTensor,
    "int16": torch.ShortTensor,
    "short": torch.ShortTensor,
    "int32": torch.IntTensor,
    "int": torch.IntTensor,
    "int64": torch.LongTensor,
    "long": torch.LongTensor,
    "bool": torch.BoolTensor,
}


class DTypeEnum(StrEnum):
    """Enum of torch dtypes."""

    float32 = auto()
    float = auto()  # float32
    float64 = auto()
    double = auto()  # float64
    float16 = auto()
    bfloat16 = auto()
    half = auto()  # float16
    uint8 = auto()  # byte
    int8 = auto()  # char
    int16 = auto()
    short = auto()  # int16
    int32 = auto()
    int = auto()  # int32
    int64 = auto()
    long = auto()  # int64
    complex32 = auto()
    chalf = auto()  # complex32
    complex64 = auto()
    cfloat = auto()  # complex64
    complex128 = auto()
    cdouble = auto()  # complex128
    quint8 = auto()
    qint8 = auto()
    qint32 = auto()
    bool = auto()
    quint4x2 = auto()
    quint2x4 = auto()

    @classmethod
    def from_dtype(cls, dtype: torch.dtype) -> "DTypeEnum":
        for name, dtype_ in TORCH_DTYPES.items():
            if dtype_ == dtype:
                return DTypeEnum.from_str(name)

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


class _TensorNDMeta(Generic[T_DType, T_NDim, TBuiltinNumber], _TensorMeta):
    def __instancecheck__(cls, instance: Any) -> bool:
        # called method to check isinstance(instance, self)
        if not isinstance(instance, torch.Tensor):
            return False

        self_generic_args = cls.__orig_class__.__args__  # type: ignore
        assert len(self_generic_args) >= 2

        self_dtype: Union[DTypeEnum, None] = self_generic_args[0].__args__[0]
        if self_dtype is not None and self_dtype.dtype != instance.dtype:
            return False

        self_ndim_type: Union[int, Type] = self_generic_args[1]
        if self_ndim_type is not int:
            meta_ndim = self_ndim_type.__args__[0]  # type: ignore
            if meta_ndim != instance.ndim:
                return False

        return True

    def __subclasscheck__(cls, subclass: Any) -> bool:
        # called method to check issubclass(subclass, cls)
        if not hasattr(subclass, "__orig_class__"):
            return False
        subcls_orig = subclass.__orig_class__
        if subcls_orig.__origin__ is not _TensorNDMeta:
            return False

        self_generic_args = cls.__orig_class__.__args__  # type: ignore
        subcls_generic_args = subcls_orig.__args__

        assert len(self_generic_args) >= 2
        assert len(subcls_generic_args) >= 2

        self_dtype = self_generic_args[0]
        subcls_dtype = subcls_generic_args[0]

        if self_dtype is not _DEFAULT_T_DTYPE and (
            subcls_dtype is _DEFAULT_T_DTYPE
            or self_dtype.__args__[0] != subcls_dtype.__args__[0]
        ):
            return False

        self_ndim = self_generic_args[1]
        subcls_ndim = subcls_generic_args[1]

        if self_ndim is not _DEFAULT_T_NDIM and (
            subcls_ndim is _DEFAULT_T_NDIM
            or self_ndim.__args__[0] != subcls_ndim.__args__[0]
        ):
            return False

        return True


class _TensorNDBase(Generic[T_DType, T_NDim, TBuiltinNumber], torch.Tensor):
    def __new__(
        cls: Type[T_Tensor],
        data: Any = None,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Device = None,
    ) -> T_Tensor:
        self_generic_args = cls.__orig_class__.__args__  # type: ignore

        # Check dtype
        self_dtype: Union[DTypeEnum, None] = self_generic_args[0].__args__[0]
        if self_dtype is None:
            pass
        elif dtype is None:
            dtype = self_dtype.dtype
        elif self_dtype.dtype != dtype:
            msg = f"Invalid argument {dtype=} for {cls.__name__}. (expected {self_dtype.dtype})"
            raise ValueError(msg)

        # Check ndim
        self_ndim_lit = self_generic_args[1]
        if self_ndim_lit is not _DEFAULT_T_NDIM:
            self_ndim: Optional[int] = self_ndim_lit.__args__[0]
        else:
            self_ndim: Optional[int] = None

        if data is not None and self_ndim is not None:
            valid, ndim = F.ndim(data, return_valid=True)
            if not valid:
                msg = f"Invalid argument data in {cls.__name__}. (cannot compute ndim for heterogeneous number of dimensions)"
                raise TypeError(msg)
            elif ndim != self_ndim:
                msg = f"Invalid number of dimension(s) for argument data in {cls.__name__}. (found {ndim} but expected {self_ndim})"
                raise ValueError(msg)

        if data is not None:
            return torch.as_tensor(data=data, dtype=dtype, device=device)  # type: ignore
        elif self_ndim is not None:
            return torch.empty([0] * self_ndim, dtype=dtype, device=device)  # type: ignore
        else:
            return torch.empty([], dtype=dtype, device=device)  # type: ignore

    ndim: T_NDim

    def item(self) -> TBuiltinNumber:
        ...

    def tolist(self) -> Union[list, TBuiltinNumber]:
        ...

    item = torch.Tensor.item  # noqa: F811  # type: ignore
    tolist = torch.Tensor.tolist  # noqa: F811


class Tensor(
    _TensorNDBase[Literal[None], int, BuiltinNumber],
    metaclass=_TensorNDMeta[Literal[None], int, BuiltinNumber],
):
    ...


class Tensor0D(
    _TensorNDBase[Literal[None], Literal[0], BuiltinNumber],
    metaclass=_TensorNDMeta[Literal[None], Literal[0], BuiltinNumber],
):
    ...


class Tensor1D(
    _TensorNDBase[Literal[None], Literal[1], BuiltinNumber],
    metaclass=_TensorNDMeta[Literal[None], Literal[1], BuiltinNumber],
):
    ...


class Tensor2D(
    _TensorNDBase[Literal[None], Literal[2], BuiltinNumber],
    metaclass=_TensorNDMeta[Literal[None], Literal[2], BuiltinNumber],
):
    ...


class Tensor3D(
    _TensorNDBase[Literal[None], Literal[3], BuiltinNumber],
    metaclass=_TensorNDMeta[Literal[None], Literal[3], BuiltinNumber],
):
    ...


class BoolTensor(
    _TensorNDBase[Literal[DTypeEnum.bool], int, bool],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], int, bool],
):
    ...


class ByteTensor(
    _TensorNDBase[Literal[DTypeEnum.uint8], int, int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], int, int],
):
    ...


class CharTensor(
    _TensorNDBase[Literal[DTypeEnum.uint8], int, int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], int, int],
):
    ...


class DoubleTensor(
    _TensorNDBase[Literal[DTypeEnum.double], int, float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], int, float],
):
    ...


class FloatTensor(
    _TensorNDBase[Literal[DTypeEnum.float], int, float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], int, float],
):
    ...


class HalfTensor(
    _TensorNDBase[Literal[DTypeEnum.half], int, float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], int, float],
):
    ...


class IntTensor(
    _TensorNDBase[Literal[DTypeEnum.int], int, int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], int, int],
):
    ...


class LongTensor(
    _TensorNDBase[Literal[DTypeEnum.long], int, int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.long], int, int],
):
    ...


class ShortTensor(
    _TensorNDBase[Literal[DTypeEnum.short], int, float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.short], int, float],
):
    ...


class BoolTensor0D(
    _TensorNDBase[Literal[DTypeEnum.bool], Literal[0], bool],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], Literal[0], bool],
):
    def tolist(self) -> bool:
        return super().tolist()  # type: ignore


class BoolTensor1D(
    _TensorNDBase[Literal[DTypeEnum.bool], Literal[1], bool],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], Literal[1], bool],
):
    def tolist(self) -> List[bool]:
        return super().tolist()  # type: ignore


class BoolTensor2D(
    _TensorNDBase[Literal[DTypeEnum.bool], Literal[2], bool],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], Literal[2], bool],
):
    def tolist(self) -> List[List[bool]]:
        return super().tolist()  # type: ignore


class BoolTensor3D(
    _TensorNDBase[Literal[DTypeEnum.bool], Literal[3], bool],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], Literal[3], bool],
):
    def tolist(self) -> List[List[List[bool]]]:
        return super().tolist()  # type: ignore


class ByteTensor0D(
    _TensorNDBase[Literal[DTypeEnum.uint8], Literal[0], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], Literal[0], int],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class ByteTensor1D(
    _TensorNDBase[Literal[DTypeEnum.uint8], Literal[1], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], Literal[1], int],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class ByteTensor2D(
    _TensorNDBase[Literal[DTypeEnum.uint8], Literal[2], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], Literal[2], int],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class ByteTensor3D(
    _TensorNDBase[Literal[DTypeEnum.uint8], Literal[3], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], Literal[3], int],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class CharTensor0D(
    _TensorNDBase[Literal[DTypeEnum.int8], Literal[0], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int8], Literal[0], int],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class CharTensor1D(
    _TensorNDBase[Literal[DTypeEnum.int8], Literal[1], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int8], Literal[1], int],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class CharTensor2D(
    _TensorNDBase[Literal[DTypeEnum.int8], Literal[2], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int8], Literal[2], int],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class CharTensor3D(
    _TensorNDBase[Literal[DTypeEnum.int8], Literal[3], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int8], Literal[3], int],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class DoubleTensor0D(
    _TensorNDBase[Literal[DTypeEnum.double], Literal[0], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], Literal[0], float],
):
    def tolist(self) -> float:
        return super().tolist()  # type: ignore


class DoubleTensor1D(
    _TensorNDBase[Literal[DTypeEnum.double], Literal[1], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], Literal[1], float],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class DoubleTensor2D(
    _TensorNDBase[Literal[DTypeEnum.double], Literal[2], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], Literal[2], float],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class DoubleTensor3D(
    _TensorNDBase[Literal[DTypeEnum.double], Literal[3], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], Literal[3], float],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class FloatTensor0D(
    _TensorNDBase[Literal[DTypeEnum.float], Literal[0], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], Literal[0], float],
):
    ...


class FloatTensor1D(
    _TensorNDBase[Literal[DTypeEnum.float], Literal[1], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], Literal[1], float],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class FloatTensor2D(
    _TensorNDBase[Literal[DTypeEnum.float], Literal[2], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], Literal[2], float],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class FloatTensor3D(
    _TensorNDBase[Literal[DTypeEnum.float], Literal[3], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], Literal[3], float],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class HalfTensor0D(
    _TensorNDBase[Literal[DTypeEnum.half], Literal[0], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], Literal[0], float],
):
    def tolist(self) -> float:
        return super().tolist()  # type: ignore


class HalfTensor1D(
    _TensorNDBase[Literal[DTypeEnum.half], Literal[1], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], Literal[1], float],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class HalfTensor2D(
    _TensorNDBase[Literal[DTypeEnum.half], Literal[2], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], Literal[2], float],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class HalfTensor3D(
    _TensorNDBase[Literal[DTypeEnum.half], Literal[3], float],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], Literal[3], float],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class IntTensor0D(
    _TensorNDBase[Literal[DTypeEnum.int], Literal[0], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[0], int],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class IntTensor1D(
    _TensorNDBase[Literal[DTypeEnum.int], Literal[1], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[1], int],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class IntTensor2D(
    _TensorNDBase[Literal[DTypeEnum.int], Literal[2], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[2], int],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class IntTensor3D(
    _TensorNDBase[Literal[DTypeEnum.int], Literal[3], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[3], int],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class LongTensor0D(
    _TensorNDBase[Literal[DTypeEnum.long], Literal[0], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.long], Literal[0], int],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class LongTensor1D(
    _TensorNDBase[Literal[DTypeEnum.long], Literal[1], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.long], Literal[1], int],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class LongTensor2D(
    _TensorNDBase[Literal[DTypeEnum.long], Literal[2], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.long], Literal[2], int],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class LongTensor3D(
    _TensorNDBase[Literal[DTypeEnum.long], Literal[3], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.long], Literal[3], int],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class ShortTensor0D(
    _TensorNDBase[Literal[DTypeEnum.short], Literal[0], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.short], Literal[0], int],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class ShortTensor1D(
    _TensorNDBase[Literal[DTypeEnum.short], Literal[1], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.short], Literal[1], int],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class ShortTensor2D(
    _TensorNDBase[Literal[DTypeEnum.short], Literal[2], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.short], Literal[2], int],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class ShortTensor3D(
    _TensorNDBase[Literal[DTypeEnum.short], Literal[3], int],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.short], Literal[3], int],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore
