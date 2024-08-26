#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import auto
from types import EllipsisType
from typing import Any, Generic, List, Literal, Type, TypeVar, Union

import torch
from torch import Tensor
from torch._C import _TensorMeta
from torch.types import _bool

from pyoutil import BuiltinNumber, StrEnum
from torchoutil.types import TORCH_DTYPES

ellipsis = EllipsisType

DType = TypeVar("DType", "DTypeEnum", None)
NDim = TypeVar("NDim", bound=int)


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

    @property
    def builtin_type(self) -> Type[BuiltinNumber]:
        return type(torch.as_tensor(0, dtype=self.dtype).item())


class _TensorNDMeta(Generic[DType, NDim], _TensorMeta):
    ndim: NDim

    def item(self) -> BuiltinNumber:
        ...

    def tolist(self) -> Union[BuiltinNumber, list]:
        ...

    def __instancecheck__(self, instance: Any) -> bool:
        if not isinstance(instance, Tensor):
            return False

        generic_args = self.__orig_class__.__args__  # type: ignore
        assert len(generic_args) >= 2

        meta_dtype: Union[DTypeEnum, None] = generic_args[0].__args__[0]
        if meta_dtype is not None and meta_dtype.dtype != instance.dtype:
            return False

        meta_ndim_type: Union[int, Type] = generic_args[1]
        if meta_ndim_type is not int:
            meta_ndim = meta_ndim_type.__args__[0]  # type: ignore
            if meta_ndim != instance.ndim:
                return False

        return True


class Tensor0D(
    Tensor,
    metaclass=_TensorNDMeta[Literal[None], Literal[0]],
):
    ...


class Tensor1D(
    Tensor,
    metaclass=_TensorNDMeta[Literal[None], Literal[1]],
):
    ...


class Tensor2D(
    Tensor,
    metaclass=_TensorNDMeta[Literal[None], Literal[2]],
):
    ...


class Tensor3D(
    Tensor,
    metaclass=_TensorNDMeta[Literal[None], Literal[3]],
):
    ...


class BoolTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.BoolTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> bool:
        return super().item()  # type: ignore


class ByteTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.ByteTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> int:
        return super().item()  # type: ignore


class CharTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.CharTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> int:
        return super().item()  # type: ignore


class DoubleTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.DoubleTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> float:
        return super().item()  # type: ignore


class FloatTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.FloatTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> float:
        return super().item()  # type: ignore


class HalfTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.HalfTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> float:
        return super().item()  # type: ignore


class IntTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.IntTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> int:
        return super().item()  # type: ignore


class LongTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.long], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.LongTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> int:
        return super().item()  # type: ignore


class ShortTensor(
    Tensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.short], int],
):
    def __init__(self, *args, **kwargs) -> None:
        torch.ShortTensor.__init__(self, *args, **kwargs)  # type: ignore

    def item(self) -> float:
        return super().item()  # type: ignore


class BoolTensor0D(
    BoolTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], Literal[0]],
):
    def tolist(self) -> bool:
        return super().tolist()  # type: ignore


class BoolTensor1D(
    BoolTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], Literal[1]],
):
    def tolist(self) -> List[bool]:
        return super().tolist()  # type: ignore


class BoolTensor2D(
    BoolTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], Literal[2]],
):
    def tolist(self) -> List[List[bool]]:
        return super().tolist()  # type: ignore


class BoolTensor3D(
    BoolTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.bool], Literal[3]],
):
    def tolist(self) -> List[List[List[bool]]]:
        return super().tolist()  # type: ignore


class ByteTensor0D(
    ByteTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], Literal[0]],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class ByteTensor1D(
    ByteTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], Literal[1]],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class ByteTensor2D(
    ByteTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], Literal[2]],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class ByteTensor3D(
    ByteTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.uint8], Literal[3]],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class CharTensor0D(
    CharTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int8], Literal[0]],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class CharTensor1D(
    CharTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int8], Literal[1]],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class CharTensor2D(
    CharTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int8], Literal[2]],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class CharTensor3D(
    CharTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int8], Literal[3]],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class DoubleTensor0D(
    DoubleTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], Literal[0]],
):
    def tolist(self) -> float:
        return super().tolist()  # type: ignore


class DoubleTensor1D(
    DoubleTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], Literal[1]],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class DoubleTensor2D(
    DoubleTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], Literal[2]],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class DoubleTensor3D(
    DoubleTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.double], Literal[3]],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class FloatTensor0D(
    FloatTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], Literal[0]],
):
    def tolist(self) -> float:
        return super().tolist()  # type: ignore


class FloatTensor1D(
    FloatTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], Literal[1]],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class FloatTensor2D(
    FloatTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], Literal[2]],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class FloatTensor3D(
    FloatTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.float], Literal[3]],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class HalfTensor0D(
    HalfTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], Literal[0]],
):
    def tolist(self) -> float:
        return super().tolist()  # type: ignore


class HalfTensor1D(
    HalfTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], Literal[1]],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class HalfTensor2D(
    HalfTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], Literal[2]],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class HalfTensor3D(
    HalfTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.half], Literal[3]],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class IntTensor0D(
    IntTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[0]],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class IntTensor1D(
    IntTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[1]],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class IntTensor2D(
    IntTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[2]],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class IntTensor3D(
    IntTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[3]],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class LongTensor0D(
    LongTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[0]],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class LongTensor1D(
    LongTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[1]],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class LongTensor2D(
    LongTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[2]],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class LongTensor3D(
    LongTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[3]],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class ShortTensor0D(
    ShortTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[0]],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class ShortTensor1D(
    ShortTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[1]],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class ShortTensor2D(
    ShortTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[2]],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class ShortTensor3D(
    ShortTensor,
    metaclass=_TensorNDMeta[Literal[DTypeEnum.int], Literal[3]],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore
