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
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
from torch._C import _TensorMeta

from torchoutil.core.dtype_enum import DTypeEnum
from torchoutil.core.get import DeviceLike, DTypeLike, get_device, get_dtype
from torchoutil.nn import functional as F
from torchoutil.pyoutil import BuiltinNumber, T_BuiltinNumber

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


class _GenericsValues(NamedTuple):
    dtype: Optional[torch.dtype] = None
    ndim: Optional[int] = None
    floating_point: Optional[bool] = None
    complex: Optional[bool] = None
    signed: Optional[bool] = None


def _get_generics(
    cls: Union[Type["_TensorNDMeta"], Type["_TensorNDBase"]],
) -> _GenericsValues:
    if not hasattr(cls, "__orig_class__"):
        return _GenericsValues()

    orig = cls.__orig_class__  # type: ignore
    if orig.__origin__ is not _TensorNDMeta:
        return _GenericsValues()

    generic_args = orig.__args__  # type: ignore
    generic_args = generic_args  # currently only check dtype and ndim
    assert len(generic_args) == 6

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

    t_floating = generic_args[3]
    if t_floating is not _DEFAULT_T_FLOATING:
        floating = t_floating.__args__[0]
    else:
        floating = None

    t_complex = generic_args[4]
    if t_complex is not _DEFAULT_T_COMPLEX:
        complex_ = t_complex.__args__[0]
    else:
        complex_ = None

    t_signed = generic_args[5]
    if t_signed is not _DEFAULT_T_SIGNED:
        signed = t_signed.__args__[0]
    else:
        signed = None

    return _GenericsValues(dtype, ndim, floating, complex_, signed)


class _TensorNDMeta(
    Generic[T_DType, T_NDim, T_BuiltinNumber, T_Floating, T_Complex, T_Signed],
    _TensorMeta,
):
    def __instancecheck__(cls, instance: Any) -> bool:
        """Called method to check isinstance(instance, self)"""
        if not isinstance(instance, torch.Tensor):
            return False

        gen = _get_generics(cls)

        if gen.dtype is not None and gen.dtype != instance.dtype:
            return False
        if gen.ndim is not None and gen.ndim != instance.ndim:
            return False
        if (
            gen.floating_point is not None
            and gen.floating_point != instance.is_floating_point()
        ):
            return False
        if gen.complex is not None and gen.complex != instance.is_complex():
            return False
        if gen.signed is not None and gen.signed != instance.is_signed():
            return False

        return True

    def __subclasscheck__(cls, subclass: Any) -> bool:
        """Called method to check issubclass(subclass, cls)"""
        self_generics = _get_generics(cls)
        other_generics = _get_generics(subclass)

        for self_attr, other_attr in zip(self_generics, other_generics):
            if self_attr is not None and (
                other_attr is None or self_attr != other_attr
            ):
                return False

        return True


class _TensorNDBase(
    Generic[T_DType, T_NDim, T_BuiltinNumber, T_Floating, T_Complex, T_Signed],
    torch.Tensor,
):
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
    ) -> T_Tensor: ...

    @overload
    def __new__(
        cls: Type[T_Tensor],
        data: Union[T_BuiltinNumber, Sequence],
        /,
        *,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
    ) -> T_Tensor: ...

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
        dtype = get_dtype(dtype)
        device = get_device(device)

        gen = _get_generics(cls)  # type: ignore
        cls_dtype = gen.dtype
        cls_ndim = gen.ndim

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

        if layout is None:  # supports older torch versions
            layout = torch.strided

        if data is not None:
            return torch.as_tensor(
                data=data,
                dtype=dtype,
                device=device,
            )  # type: ignore
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
    ) -> None: ...

    @overload
    def __init__(
        self,
        data: Union[T_BuiltinNumber, Sequence],
        /,
        *,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
    ) -> None: ...

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
    ) -> None: ...

    ndim: T_NDim  # type: ignore

    def item(self) -> T_BuiltinNumber:  # type: ignore
        ...

    def tolist(self) -> Union[list, T_BuiltinNumber]:  # type: ignore
        ...

    def is_floating_point(self) -> T_Floating:  # type: ignore
        ...

    def is_complex(self) -> T_Complex:  # type: ignore
        ...

    def is_signed(self) -> T_Signed:  # type: ignore
        ...

    item = torch.Tensor.item  # noqa: F811  # type: ignore
    tolist = torch.Tensor.tolist  # noqa: F811
    is_floating_point = torch.Tensor.is_floating_point  # noqa: F811
    is_complex = torch.Tensor.is_complex  # noqa: F811
    is_signed = torch.Tensor.is_signed  # noqa: F811


class Tensor(
    _TensorNDBase[Literal[None], int, BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], int, BuiltinNumber, bool, bool, bool],
): ...


class Tensor0D(
    _TensorNDBase[Literal[None], Literal[0], BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], Literal[0], BuiltinNumber, bool, bool, bool],
): ...


class Tensor1D(
    _TensorNDBase[Literal[None], Literal[1], BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], Literal[1], BuiltinNumber, bool, bool, bool],
): ...


class Tensor2D(
    _TensorNDBase[Literal[None], Literal[2], BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], Literal[2], BuiltinNumber, bool, bool, bool],
): ...


class Tensor3D(
    _TensorNDBase[Literal[None], Literal[3], BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], Literal[3], BuiltinNumber, bool, bool, bool],
): ...


class BoolTensor(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        int,
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        int,
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
): ...


class BoolTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[0],
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        Literal[0],
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> bool:
        return super().tolist()  # type: ignore


class BoolTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[1],
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        Literal[1],
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[bool]:
        return super().tolist()  # type: ignore


class BoolTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[2],
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        Literal[2],
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[List[bool]]:
        return super().tolist()  # type: ignore


class BoolTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[3],
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        Literal[3],
        bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[List[List[bool]]]:
        return super().tolist()  # type: ignore


class ByteTensor(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
): ...


class ByteTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class ByteTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class ByteTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class ByteTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class CharTensor(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
): ...


class CharTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class CharTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class CharTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class CharTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class DoubleTensor(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        int,
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        int,
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
): ...


class DoubleTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[0],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        Literal[0],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> float:
        return super().tolist()  # type: ignore


class DoubleTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[1],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        Literal[1],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class DoubleTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[2],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        Literal[2],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class DoubleTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[3],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        Literal[3],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class FloatTensor(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        int,
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        int,
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
): ...


class FloatTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[0],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        Literal[0],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
): ...


class FloatTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[1],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        Literal[1],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class FloatTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[2],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        Literal[2],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class FloatTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[3],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        Literal[3],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class HalfTensor(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        int,
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        int,
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
): ...


class HalfTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[0],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        Literal[0],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> float:
        return super().tolist()  # type: ignore


class HalfTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[1],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        Literal[1],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[float]:
        return super().tolist()  # type: ignore


class HalfTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[2],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        Literal[2],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[float]]:
        return super().tolist()  # type: ignore


class HalfTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[3],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        Literal[3],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[float]]]:
        return super().tolist()  # type: ignore


class IntTensor(
    _TensorNDBase[
        Literal[DTypeEnum.int], int, int, Literal[False], Literal[False], Literal[True]
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int], int, int, Literal[False], Literal[False], Literal[True]
    ],
): ...


class IntTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class IntTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class IntTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class IntTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class LongTensor(
    _TensorNDBase[
        Literal[DTypeEnum.long], int, int, Literal[False], Literal[False], Literal[True]
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long], int, int, Literal[False], Literal[False], Literal[True]
    ],
): ...


class LongTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class LongTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class LongTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class LongTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class ShortTensor(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
): ...


class ShortTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        Literal[0],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> int:
        return super().tolist()  # type: ignore


class ShortTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        Literal[1],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[int]:
        return super().tolist()  # type: ignore


class ShortTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        Literal[2],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[int]]:
        return super().tolist()  # type: ignore


class ShortTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[int]]]:
        return super().tolist()  # type: ignore


class CFloatTensor(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cfloat],
        int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
): ...


class CFloatTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cfloat],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    def tolist(self) -> complex:
        return super().tolist()  # type: ignore


class CFloatTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cfloat],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    def tolist(self) -> List[complex]:
        return super().tolist()  # type: ignore


class CFloatTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cfloat],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[complex]]:
        return super().tolist()  # type: ignore


class CFloatTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cfloat],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[complex]]]:
        return super().tolist()  # type: ignore
