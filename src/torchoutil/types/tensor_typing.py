#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tensor subclasses for typing and instance checks.

Note: torchoutil.FloatTensor != torch.FloatTensor but issubclass(torchoutil.FloatTensor, torch.FloatTensor) is False because torch.FloatTensor cannot be subclassed
"""

from typing import (
    Any,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
from torch._C import _TensorMeta

from torchoutil.nn import functional as F
from torchoutil.pyoutil import BuiltinNumber, T_BuiltinNumber
from torchoutil.types.classes import DeviceLike, DTypeLike
from torchoutil.types.dtype_typing import DTypeEnum

T_Tensor = TypeVar("T_Tensor", bound=torch.Tensor)
T_DType = TypeVar("T_DType", "DTypeEnum", None)
T_NDim = TypeVar("T_NDim", bound=int)
T_Floating = TypeVar("T_Floating", bound=bool)
T_Complex = TypeVar("T_Complex", bound=bool)
T_Signed = TypeVar("T_Signed", bound=bool)

_DEFAULT_T_DTYPE = Literal[None]
_DEFAULT_T_NDIM = int
_DEFAULT_T_FLOATING = bool
_DEFAULT_T_COMPLEX = bool
_DEFAULT_T_SIGNED = bool

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


def _get_generics(
    cls: Union[Type["_TensorNDMeta"], Type["_TensorNDBase"]],
) -> Tuple[Optional[torch.dtype], Optional[int]]:
    if not hasattr(cls, "__orig_class__"):
        return None, None
    orig = cls.__orig_class__  # type: ignore
    if orig.__origin__ is not _TensorNDMeta:
        return None, None

    generic_args = orig.__args__  # type: ignore
    assert len(generic_args) >= 2

    t_dtype = generic_args[0]
    if t_dtype is not _DEFAULT_T_DTYPE:
        enum_dtype: DTypeEnum = t_dtype.__args__[0]
        dtype = enum_dtype.dtype
    else:
        dtype = None

    t_ndim = generic_args[1]
    if t_ndim is not _DEFAULT_T_NDIM:
        ndim = t_ndim.__args__[0]
    else:
        ndim = None

    return dtype, ndim


class _TensorNDMeta(Generic[T_DType, T_NDim, T_BuiltinNumber], _TensorMeta):
    def __instancecheck__(cls, instance: Any) -> bool:
        """Called method to check isinstance(instance, self)"""
        if not isinstance(instance, torch.Tensor):
            return False

        dtype, ndim = _get_generics(cls)

        if dtype is not None and dtype != instance.dtype:
            return False
        if ndim is not None and ndim != instance.ndim:
            return False

        return True

    def __subclasscheck__(cls, subclass: Any) -> bool:
        """Called method to check issubclass(subclass, cls)"""
        self_dtype, self_ndim = _get_generics(cls)
        other_dtype, other_ndim = _get_generics(subclass)

        if self_dtype is not None and (
            other_dtype is None or self_dtype != other_dtype
        ):
            return False

        if self_ndim is not None and (other_ndim is None or self_ndim != other_ndim):
            return False

        return True


class _TensorNDBase(Generic[T_DType, T_NDim, T_BuiltinNumber], torch.Tensor):
    @overload
    def __new__(
        cls: Type[T_Tensor],
        *dims: int,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: Union[torch.memory_format, None] = None,
        out: Union[torch.Tensor, None] = None,
        layout: Union[torch.layout, None] = None,
        pin_memory: Union[bool, None] = False,
        requires_grad: Union[bool, None] = False,
    ) -> T_Tensor:
        ...

    @overload
    def __new__(
        cls: Type[T_Tensor],
        data: Union[T_BuiltinNumber, Sequence],
        /,
        *,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
    ) -> T_Tensor:
        ...

    def __new__(
        cls: Type[T_Tensor],
        *args: Any,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: Union[torch.memory_format, None] = None,
        out: Union[torch.Tensor, None] = None,
        layout: Union[torch.layout, None] = None,
        pin_memory: Union[bool, None] = False,
        requires_grad: Union[bool, None] = False,
    ) -> T_Tensor:
        dtype = F.get_dtype(dtype)
        device = F.get_device(device)
        cls_dtype, cls_ndim = _get_generics(cls)  # type: ignore

        if cls_dtype is None:
            pass
        elif dtype is None:
            dtype = cls_dtype
        elif cls_dtype != dtype:
            msg = (
                f"Invalid argument {dtype=} for {cls.__name__}. (expected {cls_dtype})"
            )
            raise ValueError(msg)

        # Sanity checks
        is_int_args = all(isinstance(arg, int) for arg in args)
        if cls_ndim is None:
            if len(args) == 0:
                size = (0,)
                data = None
            elif is_int_args:
                size = args
                data = None
            elif len(args) == 1 and not isinstance(args[0], int):
                size = None
                data = args[0]
            else:
                msg = f"Invalid arguments {args=}. (expected only ints or one sequence of data)"
                raise ValueError(msg)

        elif cls_ndim == len(args) and is_int_args:
            size = args
            data = None
        elif len(args) == 1:
            size = None
            data = args[0]
        elif len(args) == 0:
            size = [0] * cls_ndim
            data = None
        else:
            msg = f"Invalid arguments {args=}. (expected {cls_ndim} ints or one sequence of data)"
            raise ValueError(msg)
        del args

        if data is not None and cls_ndim is not None:
            valid, ndim = F.ndim(data, return_valid=True)
            if not valid:
                msg = f"Invalid argument data in {cls.__name__}. (cannot compute ndim for heterogeneous number of dimensions)"
                raise TypeError(msg)
            elif ndim != cls_ndim:
                msg = f"Invalid number of dimension(s) for argument data in {cls.__name__}. (found {ndim} but expected {cls_ndim})"
                raise ValueError(msg)

        if data is not None:
            return torch.as_tensor(data=data, dtype=dtype, device=device)  # type: ignore
        elif size is not None:
            return torch.empty(
                size,
                dtype=dtype,
                device=device,
                memory_format=memory_format,
                out=out,
                layout=layout,
                pin_memory=pin_memory,
                requires_grad=requires_grad,
            )  # type: ignore
        else:
            msg = f"Internal error: found {data=} and {size=} in {cls.__name__}."
            raise RuntimeError(msg)

    @overload
    def __init__(
        self,
        *dims: int,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: Union[torch.memory_format, None] = None,
        out: Union[torch.Tensor, None] = None,
        layout: Union[torch.layout, None] = None,
        pin_memory: Union[bool, None] = False,
        requires_grad: Union[bool, None] = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        data: Union[T_BuiltinNumber, Sequence],
        /,
        *,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
    ) -> None:
        ...

    def __init__(
        self,
        *args: Any,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: Union[torch.memory_format, None] = None,
        out: Union[torch.Tensor, None] = None,
        layout: Union[torch.layout, None] = None,
        pin_memory: Union[bool, None] = False,
        requires_grad: Union[bool, None] = False,
    ) -> None:
        ...

    ndim: T_NDim

    def item(self) -> T_BuiltinNumber:
        ...

    def tolist(self) -> Union[list, T_BuiltinNumber]:
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


class CFloatTensor0D(
    _TensorNDBase[Literal[DTypeEnum.cfloat], Literal[0], complex],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.cfloat], Literal[0], complex],
):
    def tolist(self) -> complex:
        return super().tolist()  # type: ignore


class CFloatTensor1D(
    _TensorNDBase[Literal[DTypeEnum.cfloat], Literal[1], complex],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.cfloat], Literal[1], complex],
):
    def tolist(self) -> List[complex]:
        return super().tolist()  # type: ignore


class CFloatTensor2D(
    _TensorNDBase[Literal[DTypeEnum.cfloat], Literal[2], complex],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.cfloat], Literal[2], complex],
):
    def tolist(self) -> List[List[complex]]:
        return super().tolist()  # type: ignore


class CFloatTensor3D(
    _TensorNDBase[Literal[DTypeEnum.cfloat], Literal[3], complex],
    metaclass=_TensorNDMeta[Literal[DTypeEnum.cfloat], Literal[3], complex],
):
    def tolist(self) -> List[List[List[complex]]]:
        return super().tolist()  # type: ignore
