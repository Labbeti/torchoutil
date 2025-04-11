#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tensor subclasses for typing and instance checks.

Note: torchoutil.FloatTensor != torch.FloatTensor but issubclass(torchoutil.FloatTensor, torch.FloatTensor) is False because torch.FloatTensor cannot be subclassed

Here is an overview of the valid tensor subclasses tree:
                                                                            Tensor
                                                                              |
                  +---------------------------------------+-------------------+------------------------------------+
                  |                                       |                                                        |
        ComplexFloatingTensor                       FloatingTensor                                           IntegralTensor
                  |                                       |                                                        |
     +------------+------------+              +-----------+-----------+                         +------------------+------------------+
     |            |            |              |           |           |                         |                                     |
CHalfTensor CFloatTensor CDoubleTensor    HalfTensor FloatTensor DoubleTensor          SignedIntegerTensor                  UnsignedIntegerTensor
   (c32)        (c64)       (c128)          (f16)       (f32)       (f64)                       |                                     |
                                                                              +-----------+-----+-----+-----------+             +-----+-----+
                                                                              |           |           |           |             |           |
                                                                          CharTensor  ShortTensor  IntTensor  LongTensor   ByteTensor   BoolTensor
                                                                             (i8)       (i16)       (i32)       (i64)         (u8)       (bool)

"""

from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

import torch
from torch._C import _TensorMeta
from torch.types import Device, _bool, _float, _int
from typing_extensions import TypeVar

from torchoutil.core.dtype_enum import DTypeEnum
from torchoutil.core.make import DeviceLike, DTypeLike, as_device, as_dtype
from torchoutil.pyoutil import BuiltinNumber, T_BuiltinNumber

_DEFAULT_T_DTYPE = Literal[None]
_DEFAULT_T_NDIM = _int
_DEFAULT_T_FLOATING = _bool
_DEFAULT_T_COMPLEX = _bool
_DEFAULT_T_SIGNED = _bool

T_Tensor = TypeVar("T_Tensor", bound="_TensorNDBase")
T_DType = TypeVar("T_DType", "DTypeEnum", None)
T_NDim = TypeVar("T_NDim", bound=_int)
T_Floating = TypeVar("T_Floating", bound=_bool)
T_Complex = TypeVar("T_Complex", bound=_bool)
T_Signed = TypeVar("T_Signed", bound=_bool)

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
    "byte": torch.ByteTensor,
    "char": torch.CharTensor,
    "int8": torch.CharTensor,
}


class _GenericsValues(NamedTuple):
    dtype: Optional[torch.dtype] = None
    ndim: Optional[_int] = None
    is_floating_point: Optional[_bool] = None
    is_complex: Optional[_bool] = None
    is_signed: Optional[_bool] = None

    def is_compatible_with_tensor(self, tensor: torch.Tensor) -> _bool:
        if self.ndim is not None and self.ndim != tensor.ndim:
            return False
        else:
            return self.is_compatible_with_dtype(tensor.dtype)

    def is_compatible_with_dtype(self, dtype: torch.dtype) -> _bool:
        if self.dtype is not None:
            return self.dtype == dtype

        for self_attr, dtype_attr in zip(
            self[2:5], (dtype.is_floating_point, dtype.is_complex, dtype.is_signed)
        ):
            if self_attr is not None and self_attr != dtype_attr:
                return False
        return True

    def is_compatible_with_generic(self, other: "_GenericsValues") -> _bool:
        for self_attr, other_attr in zip(self, other):
            if self_attr is not None and (
                other_attr is None or self_attr != other_attr
            ):
                return False
        return True


def _get_generics(
    cls: Union["_TensorNDMeta", "_TensorNDBase"],
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
    """Tensor metaclass with redefined instance check based on generic properties."""

    def __instancecheck__(self, instance: Any) -> _bool:
        """Called method to check isinstance(instance, self)"""
        if not isinstance(instance, torch.Tensor):
            return False

        gen = _get_generics(self)
        return gen.is_compatible_with_tensor(instance)

    def __subclasscheck__(self, subclass: Any) -> _bool:
        """Called method to check issubclass(subclass, cls)"""
        self_generics = _get_generics(self)
        other_generics = _get_generics(subclass)
        return self_generics.is_compatible_with_generic(other_generics)


class _TensorNDBase(
    Generic[T_DType, T_NDim, T_BuiltinNumber, T_Floating, T_Complex, T_Signed],
    torch.Tensor,
):
    """Tensor base class with redefined constructor and overloaded methods."""

    _DEFAULT_DTYPE: ClassVar[Optional[DTypeEnum]] = None

    @overload
    def __new__(
        cls: Type[T_Tensor],
        *dims: _int,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: Union[torch.memory_format, None] = None,
        out: Union[torch.Tensor, None] = None,
        layout: Union[torch.layout, None] = None,
        pin_memory: Union[_bool, None] = False,
        requires_grad: Union[_bool, None] = False,
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
        pin_memory: Union[_bool, None] = False,
        requires_grad: Union[_bool, None] = False,
    ) -> T_Tensor:
        dtype = as_dtype(dtype)
        device = as_device(device)

        gen = _get_generics(cls)  # type: ignore
        cls_dtype = gen.dtype
        cls_ndim = gen.ndim

        # Sanity checks for dtype
        if dtype is None:
            if cls_dtype is not None:
                dtype = cls_dtype
            elif cls._DEFAULT_DTYPE is not None:
                dtype = cls._DEFAULT_DTYPE.dtype

        elif cls_dtype is None:
            if not gen.is_compatible_with_dtype(dtype):
                msg = f"Invalid argument {dtype=} for {cls.__name__}. (expected a dtype with (is_floating_point={gen.is_floating_point}, is_complex={gen.is_complex}, is_signed={gen.is_signed}))"
                raise ValueError(msg)

        elif dtype == cls_dtype:
            pass
        else:
            msg = (
                f"Invalid argument {dtype=} for {cls.__name__}. (expected {cls_dtype})"
            )
            raise ValueError(msg)

        # Sanity checks for data and ndim
        is_int_args = all(isinstance(arg, _int) for arg in args)
        if cls_ndim is None:
            if len(args) == 0:
                size = (0,)
                data = None
            elif is_int_args:
                size = args
                data = None
            elif len(args) == 1 and not isinstance(args[0], _int):
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

        # Supports older PyTorch versions
        if layout is None:
            layout = torch.strided
        if pin_memory is None:
            pin_memory = False
        if requires_grad is None:
            requires_grad = False

        if data is not None:
            result = torch.as_tensor(
                data=data,
                dtype=dtype,  # type: ignore
                device=device,
            )
            if cls_ndim is not None and result.ndim != cls_ndim:
                msg = f"Invalid number of dimension(s) for argument data in {cls.__name__}. (found {result.ndim} but expected {cls_ndim})"
                raise ValueError(msg)
            return result  # type: ignore

        elif size is not None:
            return torch.empty(
                size,
                dtype=dtype,  # type: ignore
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
        *dims: _int,
        dtype: DTypeLike = None,
        device: DeviceLike = None,
        memory_format: Union[torch.memory_format, None] = None,
        out: Union[torch.Tensor, None] = None,
        layout: Union[torch.layout, None] = None,
        pin_memory: Union[_bool, None] = False,
        requires_grad: Union[_bool, None] = False,
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
        pin_memory: Union[_bool, None] = False,
        requires_grad: Union[_bool, None] = False,
    ) -> None:
        ...

    @overload
    def __eq__(self, other: Any) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def __getitem__(self: "Tensor1D", idx: _int, /) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def __getitem__(self: "Tensor2D", idx: _int, /) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def __getitem__(self: "Tensor3D", idx: _int, /) -> "Tensor2D":  # type: ignore
        ...

    @overload
    def __getitem__(self: "Tensor0D", idx: None, /) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def __getitem__(self: "Tensor1D", idx: None, /) -> "Tensor2D":  # type: ignore
        ...

    @overload
    def __getitem__(self: "Tensor2D", idx: None, /) -> "Tensor3D":  # type: ignore
        ...

    @overload
    def __getitem__(self: "Tensor3D", idx: None, /) -> "Tensor":  # type: ignore
        ...

    @overload
    def __getitem__(self: T_Tensor, sl: slice, /) -> T_Tensor:
        ...

    @overload
    def __getitem__(self, *args) -> "Tensor":
        ...

    @overload
    def __ne__(self, other: Any) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def abs(self: T_Tensor) -> T_Tensor:  # type: ignore
        ...

    @overload
    def absolute(self: T_Tensor) -> T_Tensor:  # type: ignore
        ...

    @overload
    def acos(self: T_Tensor) -> T_Tensor:  # type: ignore
        ...

    @overload
    def all(self, dim: Literal[None] = None) -> "BoolTensor0D":  # type: ignore
        ...

    @overload
    def all(self, dim: Union[_int, Tuple[_int, ...]], keepdim: _bool = False) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def any(self, dim: Literal[None] = None) -> "BoolTensor0D":  # type: ignore
        ...

    @overload
    def any(self, dim: Union[_int, Tuple[_int, ...]], keepdim: _bool = False) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def bool(self: T_Tensor) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def contiguous(self: T_Tensor) -> T_Tensor:  # type: ignore
        ...

    @overload
    def double(self) -> "DoubleTensor":  # type: ignore
        ...

    @overload
    def eq(self: T_Tensor, other: Union[torch.Tensor, BuiltinNumber]) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def equal(self: T_Tensor, other: torch.Tensor) -> _bool:  # type: ignore
        ...

    @overload
    def float(self) -> "FloatTensor":  # type: ignore
        ...

    @overload
    def half(self) -> "HalfTensor":  # type: ignore
        ...

    @overload
    def int(self) -> "IntTensor":  # type: ignore
        ...

    @overload
    def is_complex(self) -> T_Complex:  # type: ignore
        ...

    @overload
    def is_floating_point(self) -> T_Floating:  # type: ignore
        ...

    @overload
    def is_signed(self) -> T_Signed:  # type: ignore
        ...

    @overload
    def isfinite(self) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def isinf(self) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def isnan(self) -> "BoolTensor":  # type: ignore
        ...

    @overload
    def item(self) -> T_BuiltinNumber:  # type: ignore
        ...

    @overload
    def long(self) -> "LongTensor":  # type: ignore
        ...

    @overload
    def mean(self, dim: Literal[None] = None) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def mean(self: "Tensor0D", dim: _int) -> "Tensor0D":
        ...

    @overload
    def mean(self: "Tensor1D", dim: _int) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def mean(self: "Tensor2D", dim: _int) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def mean(self: "Tensor3D", dim: _int) -> "Tensor2D":  # type: ignore
        ...

    @overload
    def mean(self, dim: _int) -> "Tensor":  # type: ignore
        ...

    @overload
    def reshape(self, size: Tuple[()]) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def reshape(self, size: Tuple[_int]) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def reshape(self, size: Tuple[_int, _int]) -> "Tensor2D":
        ...

    @overload
    def reshape(self, size: Tuple[_int, _int, _int]) -> "Tensor3D":
        ...

    @overload
    def reshape(self, size: Tuple[_int, ...]) -> "Tensor":
        ...

    @overload
    def reshape(self, size0: _int) -> "Tensor1D":
        ...

    @overload
    def reshape(self, size0: _int, size1: _int) -> "Tensor2D":
        ...

    @overload
    def reshape(self, size0: _int, size1: _int, size2: _int) -> "Tensor3D":  # type: ignore
        ...

    @overload
    def short(self) -> "ShortTensor":  # type: ignore
        ...

    @overload
    def squeeze(self, dim: Optional[_int] = None) -> "Tensor":  # type: ignore
        ...

    @overload
    def sum(self, dim: Literal[None] = None) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def sum(self: "Tensor1D", dim: _int) -> "Tensor0D":
        ...

    @overload
    def sum(self: "Tensor2D", dim: _int) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def sum(self: "Tensor3D", dim: _int) -> "Tensor2D":  # type: ignore
        ...

    @overload
    def sum(self, dim: Optional[_int] = None) -> "Tensor":  # type: ignore
        ...

    @overload
    def to(  # type: ignore
        self: T_Tensor,
        dtype: Optional[torch.dtype] = None,
        non_blocking: _bool = False,
        copy: _bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> T_Tensor:
        ...

    @overload
    def to(
        self: T_Tensor,
        device: Device = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: _bool = False,
        copy: _bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> T_Tensor:
        ...

    @overload
    def to(
        self,
        other: T_Tensor,
        non_blocking: _bool = False,
        copy: _bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> T_Tensor:
        ...

    @overload
    def tolist(self) -> Union[list, T_BuiltinNumber]:  # type: ignore
        ...

    @overload
    def unsqueeze(self: "Tensor0D", dim: _int) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def unsqueeze(self: "Tensor1D", dim: _int) -> "Tensor2D":  # type: ignore
        ...

    @overload
    def unsqueeze(self: "Tensor2D", dim: _int) -> "Tensor3D":  # type: ignore
        ...

    @overload
    def unsqueeze(self, dim: _int) -> "Tensor":  # type: ignore
        ...

    @overload
    def view(self, size: Tuple[()]) -> "Tensor0D":  # type: ignore
        ...

    @overload
    def view(self, size: Tuple[_int]) -> "Tensor1D":  # type: ignore
        ...

    @overload
    def view(self, size: Tuple[_int, _int]) -> "Tensor2D":
        ...

    @overload
    def view(self, size: Tuple[_int, _int, _int]) -> "Tensor3D":
        ...

    @overload
    def view(self, size: Tuple[_int, ...]) -> "Tensor":
        ...

    @overload
    def view(self, size0: _int) -> "Tensor1D":
        ...

    @overload
    def view(self, size0: _int, size1: _int) -> "Tensor2D":
        ...

    @overload
    def view(self, size0: _int, size1: _int, size2: _int) -> "Tensor3D":
        ...

    @overload
    def view(self, *size: _int) -> "Tensor":  # type: ignore
        ...

    @overload
    def view(self, dtype: torch.dtype) -> "Tensor":  # type: ignore
        ...

    ndim: T_NDim  # type: ignore

    __eq__ = torch.Tensor.__eq__  # noqa: F811  # type: ignore
    __getitem__ = torch.Tensor.__getitem__  # noqa: F811  # type: ignore
    __ne__ = torch.Tensor.__ne__  # noqa: F811  # type: ignore
    abs = torch.Tensor.abs  # noqa: F811  # type: ignore
    absolute = torch.Tensor.absolute  # noqa: F811  # type: ignore
    acos = torch.Tensor.acos  # noqa: F811  # type: ignore
    all = torch.Tensor.all  # noqa: F811  # type: ignore
    any = torch.Tensor.any  # noqa: F811  # type: ignore
    bool = torch.Tensor.bool  # noqa: F811  # type: ignore
    contiguous = torch.Tensor.contiguous  # noqa: F811  # type: ignore
    double = torch.Tensor.double  # noqa: F811  # type: ignore
    eq = torch.Tensor.eq  # noqa: F811  # type: ignore
    equal = torch.Tensor.equal  # noqa: F811  # type: ignore
    float = torch.Tensor.float  # noqa: F811  # type: ignore
    half = torch.Tensor.half  # noqa: F811  # type: ignore
    is_complex = torch.Tensor.is_complex  # noqa: F811  # type: ignore
    is_floating_point = torch.Tensor.is_floating_point  # noqa: F811  # type: ignore
    is_signed = torch.Tensor.is_signed  # noqa: F811  # type: ignore
    isfinite = torch.Tensor.isfinite  # noqa: F811  # type: ignore
    isinf = torch.Tensor.isinf  # noqa: F811  # type: ignore
    isnan = torch.Tensor.isnan  # noqa: F811  # type: ignore
    int = torch.Tensor.int  # noqa: F811  # type: ignore
    item = torch.Tensor.item  # noqa: F811  # type: ignore
    long = torch.Tensor.long  # noqa: F811  # type: ignore
    mean = torch.Tensor.mean  # noqa: F811  # type: ignore
    reshape = torch.Tensor.reshape  # noqa: F811  # type: ignore
    short = torch.Tensor.short  # noqa: F811  # type: ignore
    squeeze = torch.Tensor.squeeze  # noqa: F811  # type: ignore
    sum = torch.Tensor.sum  # noqa: F811  # type: ignore
    to = torch.Tensor.to  # noqa: F811  # type: ignore
    tolist = torch.Tensor.tolist  # noqa: F811  # type: ignore
    unsqueeze = torch.Tensor.unsqueeze  # noqa: F811  # type: ignore
    view = torch.Tensor.view  # noqa: F811  # type: ignore


class Tensor(
    _TensorNDBase[Literal[None], _int, BuiltinNumber, _bool, _bool, _bool],
    metaclass=_TensorNDMeta[Literal[None], _int, BuiltinNumber, _bool, _bool, _bool],
):
    ...


class Tensor0D(
    _TensorNDBase[Literal[None], Literal[0], BuiltinNumber, _bool, _bool, _bool],
    metaclass=_TensorNDMeta[
        Literal[None], Literal[0], BuiltinNumber, _bool, _bool, _bool
    ],
):
    ...


class Tensor1D(
    _TensorNDBase[Literal[None], Literal[1], BuiltinNumber, _bool, _bool, _bool],
    metaclass=_TensorNDMeta[
        Literal[None], Literal[1], BuiltinNumber, _bool, _bool, _bool
    ],
):
    ...


class Tensor2D(
    _TensorNDBase[Literal[None], Literal[2], BuiltinNumber, _bool, _bool, _bool],
    metaclass=_TensorNDMeta[
        Literal[None], Literal[2], BuiltinNumber, _bool, _bool, _bool
    ],
):
    ...


class Tensor3D(
    _TensorNDBase[Literal[None], Literal[3], BuiltinNumber, _bool, _bool, _bool],
    metaclass=_TensorNDMeta[
        Literal[None], Literal[3], BuiltinNumber, _bool, _bool, _bool
    ],
):
    ...


# ----------------------------------------
# Concrete classes
# ----------------------------------------


class BoolTensor(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        _int,
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        _int,
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    ...


class BoolTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[0],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        Literal[0],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> _bool:
        return super().tolist()  # type: ignore


class BoolTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[1],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        Literal[1],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[_bool]:
        return super().tolist()  # type: ignore


class BoolTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[2],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        Literal[2],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[List[_bool]]:
        return super().tolist()  # type: ignore


class BoolTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.bool],
        Literal[3],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.bool],
        Literal[3],
        _bool,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[List[List[_bool]]]:
        return super().tolist()  # type: ignore


class ByteTensor(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    ...


class ByteTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> _int:
        return super().tolist()  # type: ignore


class ByteTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[_int]:
        return super().tolist()  # type: ignore


class ByteTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[List[_int]]:
        return super().tolist()  # type: ignore


class ByteTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.uint8],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.uint8],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    def tolist(self) -> List[List[List[_int]]]:
        return super().tolist()  # type: ignore


class CharTensor(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    ...


class CharTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> _int:
        return super().tolist()  # type: ignore


class CharTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[_int]:
        return super().tolist()  # type: ignore


class CharTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[_int]]:
        return super().tolist()  # type: ignore


class CharTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.int8],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[_int]]]:
        return super().tolist()  # type: ignore


class DoubleTensor(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    ...


class DoubleTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> _float:
        return super().tolist()  # type: ignore


class DoubleTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[_float]:
        return super().tolist()  # type: ignore


class DoubleTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[_float]]:
        return super().tolist()  # type: ignore


class DoubleTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.double],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.double],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[_float]]]:
        return super().tolist()  # type: ignore


class FloatTensor(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    ...


class FloatTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    ...


class FloatTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[_float]:
        return super().tolist()  # type: ignore


class FloatTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[_float]]:
        return super().tolist()  # type: ignore


class FloatTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.float],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.float],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[_float]]]:
        return super().tolist()  # type: ignore


class HalfTensor(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    ...


class HalfTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> _float:
        return super().tolist()  # type: ignore


class HalfTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[_float]:
        return super().tolist()  # type: ignore


class HalfTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[_float]]:
        return super().tolist()  # type: ignore


class HalfTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.half],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.half],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[_float]]]:
        return super().tolist()  # type: ignore


class IntTensor(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    ...


class IntTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> _int:
        return super().tolist()  # type: ignore


class IntTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[_int]:
        return super().tolist()  # type: ignore


class IntTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[_int]]:
        return super().tolist()  # type: ignore


class IntTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.int],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[_int]]]:
        return super().tolist()  # type: ignore


class LongTensor(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    ...


class LongTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> _int:
        return super().tolist()  # type: ignore


class LongTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[_int]:
        return super().tolist()  # type: ignore


class LongTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[_int]]:
        return super().tolist()  # type: ignore


class LongTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.long],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.long],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[_int]]]:
        return super().tolist()  # type: ignore


class ShortTensor(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    ...


class ShortTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> _int:
        return super().tolist()  # type: ignore


class ShortTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[_int]:
        return super().tolist()  # type: ignore


class ShortTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[_int]]:
        return super().tolist()  # type: ignore


class ShortTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.short],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.short],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[_int]]]:
        return super().tolist()  # type: ignore


class CFloatTensor(
    _TensorNDBase[
        Literal[DTypeEnum.cfloat],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cfloat],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    ...


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


if hasattr(torch, "complex32"):

    class CHalfTensor(
        _TensorNDBase[
            Literal[DTypeEnum.chalf],
            _int,
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
        metaclass=_TensorNDMeta[
            Literal[DTypeEnum.chalf],
            _int,
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
    ):
        ...

    class CHalfTensor0D(
        _TensorNDBase[
            Literal[DTypeEnum.chalf],
            Literal[0],
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
        metaclass=_TensorNDMeta[
            Literal[DTypeEnum.chalf],
            Literal[0],
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
    ):
        def tolist(self) -> complex:
            return super().tolist()  # type: ignore

    class CHalfTensor1D(
        _TensorNDBase[
            Literal[DTypeEnum.chalf],
            Literal[1],
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
        metaclass=_TensorNDMeta[
            Literal[DTypeEnum.chalf],
            Literal[1],
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
    ):
        def tolist(self) -> List[complex]:
            return super().tolist()  # type: ignore

    class CHalfTensor2D(
        _TensorNDBase[
            Literal[DTypeEnum.chalf],
            Literal[2],
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
        metaclass=_TensorNDMeta[
            Literal[DTypeEnum.chalf],
            Literal[2],
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
    ):
        def tolist(self) -> List[List[complex]]:
            return super().tolist()  # type: ignore

    class CHalfTensor3D(
        _TensorNDBase[
            Literal[DTypeEnum.chalf],
            Literal[3],
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
        metaclass=_TensorNDMeta[
            Literal[DTypeEnum.chalf],
            Literal[3],
            complex,
            Literal[False],
            Literal[True],
            Literal[True],
        ],
    ):
        def tolist(self) -> List[List[List[complex]]]:
            return super().tolist()  # type: ignore


class CDoubleTensor(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cdouble],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    ...


class CDoubleTensor0D(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cdouble],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    def tolist(self) -> complex:
        return super().tolist()  # type: ignore


class CDoubleTensor1D(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cdouble],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    def tolist(self) -> List[complex]:
        return super().tolist()  # type: ignore


class CDoubleTensor2D(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cdouble],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[complex]]:
        return super().tolist()  # type: ignore


class CDoubleTensor3D(
    _TensorNDBase[
        Literal[DTypeEnum.cdouble],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cdouble],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    def tolist(self) -> List[List[List[complex]]]:
        return super().tolist()  # type: ignore


# ----------------------------------------
# Intermediate classes
# ----------------------------------------


class ComplexFloatingTensor(
    _TensorNDBase[
        Literal[None],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        _int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    """Intermediate class for checking and typing complex-valued tensors.
    - Concrete subclasses are: CFloatTensor, CHalfTensor, CDoubleTensor.
    - Properties are: is_floating_point=False, is_complex=True, is_signed=True.
    - By default, instantiate this class will create a CFloatTensor.
    """

    _DEFAULT_DTYPE = DTypeEnum.complex64


class ComplexFloatingTensor0D(
    _TensorNDBase[
        Literal[None],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[0],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.complex64


class ComplexFloatingTensor1D(
    _TensorNDBase[
        Literal[None],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[1],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.complex64


class ComplexFloatingTensor2D(
    _TensorNDBase[
        Literal[None],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[2],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.complex64


class ComplexFloatingTensor3D(
    _TensorNDBase[
        Literal[None],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[3],
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.complex64


class FloatingTensor(
    _TensorNDBase[
        Literal[None],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        _int,
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    """Intermediate class for checking and typing floating-point tensors.
    - Concrete subclasses are: FloatTensor, HalfTensor, DoubleTensor.
    - Properties are: is_floating_point=True, is_complex=False, is_signed=True.
    - By default, instantiate this class will create a FloatTensor.
    """

    _DEFAULT_DTYPE = DTypeEnum.float32


class FloatingTensor0D(
    _TensorNDBase[
        Literal[None],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[0],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.float32


class FloatingTensor1D(
    _TensorNDBase[
        Literal[None],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[1],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.float32


class FloatingTensor2D(
    _TensorNDBase[
        Literal[None],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[2],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.float32


class FloatingTensor3D(
    _TensorNDBase[
        Literal[None],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[3],
        _float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.float32


class SignedIntegerTensor(
    _TensorNDBase[
        Literal[None],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    """Intermediate class for checking and typing signed integer data type (integer-like) tensors.
    - Concrete subclasses are: IntTensor, LongTensor, ShortTensor.
    - Properties are: is_floating_point=False, is_complex=False, is_signed=True.
    - By default, instantiate this class will create an IntTensor.
    - BoolTensor is not a subclass of SignedIntegerTensor because it is not signed.
    """

    _DEFAULT_DTYPE = DTypeEnum.int32


class SignedIntegerTensor0D(
    _TensorNDBase[
        Literal[None],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.int32


class SignedIntegerTensor1D(
    _TensorNDBase[
        Literal[None],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.int32


class SignedIntegerTensor2D(
    _TensorNDBase[
        Literal[None],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.int32


class SignedIntegerTensor3D(
    _TensorNDBase[
        Literal[None],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.int32


class UnsignedIntegerTensor(
    _TensorNDBase[
        Literal[None],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        _int,
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    """Intermediate class for checking and typing unsigned integer data type (integer-like) tensors.
    - Concrete subclasses are: BoolTensor, ByteTensor.
    - Properties are: is_floating_point=False, is_complex=False, is_signed=False.
    - By default, instantiate this class will create an ByteTensor.
    - BoolTensor is a subclass of UnsignedIntegerTensor.
    """

    _DEFAULT_DTYPE = DTypeEnum.uint8


class UnsignedIntegerTensor0D(
    _TensorNDBase[
        Literal[None],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.uint8


class UnsignedIntegerTensor1D(
    _TensorNDBase[
        Literal[None],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.uint8


class UnsignedIntegerTensor2D(
    _TensorNDBase[
        Literal[None],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.uint8


class UnsignedIntegerTensor3D(
    _TensorNDBase[
        Literal[None],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.uint8


class IntegralTensor(
    _TensorNDBase[
        Literal[None],
        _int,
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        _int,
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
):
    """Intermediate class for checking and typing integer data type (integer-like) tensors.
    - Concrete subclasses are: CharTensor, ShortTensor, IntTensor, LongTensor, BoolTensor, ByteTensor.
    - Properties are: is_floating_point=False, is_complex=False.
    - By default, instantiate this class will create an LongTensor.
    - BoolTensor is a subclass of UnsignedIntegerTensor.
    """

    _DEFAULT_DTYPE = DTypeEnum.long


class IntegralTensor0D(
    _TensorNDBase[
        Literal[None],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[0],
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.long


class IntegralTensor1D(
    _TensorNDBase[
        Literal[None],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[1],
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.long


class IntegralTensor2D(
    _TensorNDBase[
        Literal[None],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[2],
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.long


class IntegralTensor3D(
    _TensorNDBase[
        Literal[None],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[3],
        _int,
        Literal[False],
        Literal[False],
        _bool,
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.long
