#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tensor subclasses for typing and instance checks.

Note: torchoutil.FloatTensor != torch.FloatTensor but issubclass(torchoutil.FloatTensor, torch.FloatTensor) is False because torch.FloatTensor cannot be subclassed

Here is an overview of the valid tensor subclasses tree:
                                                                            Tensor
                                                                              |
                  +---------------------------------------+-------------------+-----------------------+-------------------------------------+
                  |                                       |                                           |                                     |
        ComplexFloatingTensor                       FloatingTensor                            SignedIntegerTensor                  UnsignedIntegerTensor
                  |                                       |                                           |                                     |
     +------------+------------+              +-----------+-----------+             +-----------+-----+-----+-----------+             +-----+-----+
     |            |            |              |           |           |             |           |           |           |             |           |
CHalfTensor CFloatTensor CDoubleTensor    HalfTensor FloatTensor DoubleTensor   CharTensor  ShortTensor  IntTensor  LongTensor    ByteTensor  BoolTensor
   (c32)        (c64)       (c128)          (f16)       (f32)       (f64)          (i8)       (i16)       (i32)       (i64)          (u8)       (bool)
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
from torch.types import Device
from typing_extensions import TypeVar

from torchoutil.core.dtype_enum import DTypeEnum
from torchoutil.core.get import DeviceLike, DTypeLike, get_device, get_dtype
from torchoutil.nn import functional as F
from torchoutil.pyoutil import BuiltinNumber, T_BuiltinNumber

_DEFAULT_T_DTYPE = Literal[None]
_DEFAULT_T_NDIM = int
_DEFAULT_T_FLOATING = bool
_DEFAULT_T_COMPLEX = bool
_DEFAULT_T_SIGNED = bool

T_Tensor = TypeVar("T_Tensor", bound="_TensorNDBase")
T_DType = TypeVar("T_DType", "DTypeEnum", None)
T_NDim = TypeVar("T_NDim", bound=int)
T_Floating = TypeVar("T_Floating", bound=bool)
T_Complex = TypeVar("T_Complex", bound=bool)
T_Signed = TypeVar("T_Signed", bound=bool)

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
    is_floating_point: Optional[bool] = None
    is_complex: Optional[bool] = None
    is_signed: Optional[bool] = None

    def is_compatible_with_tensor(self, tensor: torch.Tensor) -> bool:
        if self.ndim is not None and self.ndim != tensor.ndim:
            return False
        else:
            return self.is_compatible_with_dtype(tensor.dtype)

    def is_compatible_with_dtype(self, dtype: torch.dtype) -> bool:
        if self.dtype is not None:
            return self.dtype == dtype

        for self_attr, dtype_attr in zip(
            self[2:5], (dtype.is_floating_point, dtype.is_complex, dtype.is_signed)
        ):
            if self_attr is not None and self_attr != dtype_attr:
                return False
        return True

    def is_compatible_with_generic(self, other: "_GenericsValues") -> bool:
        for self_attr, other_attr in zip(self, other):
            if self_attr is not None and (
                other_attr is None or self_attr != other_attr
            ):
                return False
        return True


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
        return gen.is_compatible_with_tensor(instance)

    def __subclasscheck__(cls, subclass: Any) -> bool:
        """Called method to check issubclass(subclass, cls)"""
        self_generics = _get_generics(cls)
        other_generics = _get_generics(subclass)
        return self_generics.is_compatible_with_generic(other_generics)


class _TensorNDBase(
    Generic[T_DType, T_NDim, T_BuiltinNumber, T_Floating, T_Complex, T_Signed],
    torch.Tensor,
):
    _DEFAULT_DTYPE: ClassVar[Optional[DTypeEnum]] = None

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
        dtype = get_dtype(dtype)
        device = get_device(device)

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

        if layout is None:  # supports older PyTorch versions
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

    @overload
    def __eq__(self, other: Any) -> "BoolTensor":
        ...

    @overload
    def __getitem__(self: "Tensor1D", /, idx: int) -> "Tensor0D":
        ...

    @overload
    def __getitem__(self: "Tensor2D", /, idx: int) -> "Tensor1D":
        ...

    @overload
    def __getitem__(self: "Tensor3D", /, idx: int) -> "Tensor2D":
        ...

    @overload
    def __getitem__(self: "Tensor0D", /, idx: None) -> "Tensor1D":
        ...

    @overload
    def __getitem__(self: "Tensor1D", /, idx: None) -> "Tensor2D":
        ...

    @overload
    def __getitem__(self: "Tensor2D", /, idx: None) -> "Tensor3D":
        ...

    @overload
    def __getitem__(self: "Tensor3D", /, idx: None) -> "Tensor":
        ...

    @overload
    def __getitem__(self: T_Tensor, /, sl: slice) -> T_Tensor:
        ...

    @overload
    def __getitem__(self, *args) -> "Tensor":
        ...

    @overload
    def __ne__(self, other: Any) -> "BoolTensor":
        ...

    @overload
    def abs(self: T_Tensor) -> T_Tensor:
        ...

    @overload
    def contiguous(self: T_Tensor) -> T_Tensor:
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
    def item(self) -> T_BuiltinNumber:  # type: ignore
        ...

    @overload
    def mean(self, dim: Literal[None] = None) -> "Tensor0D":
        ...

    @overload
    def mean(self: "Tensor1D", dim: int) -> "Tensor0D":
        ...

    @overload
    def mean(self: "Tensor2D", dim: int) -> "Tensor1D":
        ...

    @overload
    def mean(self: "Tensor3D", dim: int) -> "Tensor2D":
        ...

    @overload
    def mean(self, dim: Optional[int] = None) -> "Tensor":
        ...

    @overload
    def reshape(self: T_Tensor, size: Tuple[()]) -> "Tensor0D":
        ...

    @overload
    def reshape(self: T_Tensor, size: Tuple[int]) -> "Tensor1D":
        ...

    @overload
    def reshape(self: T_Tensor, size: Tuple[int, int]) -> "Tensor2D":
        ...

    @overload
    def reshape(self: T_Tensor, size: Tuple[int, int, int]) -> "Tensor3D":
        ...

    @overload
    def reshape(self: T_Tensor, size: Tuple[int, ...]) -> "Tensor":
        ...

    @overload
    def reshape(self: T_Tensor, size0: int) -> "Tensor1D":
        ...

    @overload
    def reshape(self: T_Tensor, size0: int, size1: int) -> "Tensor2D":
        ...

    @overload
    def reshape(self: T_Tensor, size0: int, size1: int, size2: int) -> "Tensor3D":
        ...

    @overload
    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        ...

    @overload
    def sum(self, dim: Literal[None] = None) -> "Tensor0D":
        ...

    @overload
    def sum(self: "Tensor1D", dim: int) -> "Tensor0D":
        ...

    @overload
    def sum(self: "Tensor2D", dim: int) -> "Tensor1D":
        ...

    @overload
    def sum(self: "Tensor3D", dim: int) -> "Tensor2D":
        ...

    @overload
    def sum(self, dim: Optional[int] = None) -> "Tensor":
        ...

    @overload
    def to(
        self: T_Tensor,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> T_Tensor:
        ...

    @overload
    def to(
        self: T_Tensor,
        device: Device = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> T_Tensor:
        ...

    @overload
    def to(
        self,
        other: T_Tensor,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: Optional[torch.memory_format] = None,
    ) -> T_Tensor:
        ...

    @overload
    def tolist(self) -> Union[list, T_BuiltinNumber]:  # type: ignore
        ...

    @overload
    def unsqueeze(self: "Tensor0D", dim: int) -> "Tensor1D":
        ...

    @overload
    def unsqueeze(self: "Tensor1D", dim: int) -> "Tensor2D":
        ...

    @overload
    def unsqueeze(self: "Tensor2D", dim: int) -> "Tensor3D":
        ...

    @overload
    def unsqueeze(self, dim: int) -> "Tensor":
        ...

    @overload
    def view(self: T_Tensor, size: Tuple[()]) -> "Tensor0D":
        ...

    @overload
    def view(self: T_Tensor, size: Tuple[int]) -> "Tensor1D":
        ...

    @overload
    def view(self: T_Tensor, size: Tuple[int, int]) -> "Tensor2D":
        ...

    @overload
    def view(self: T_Tensor, size: Tuple[int, int, int]) -> "Tensor3D":
        ...

    @overload
    def view(self: T_Tensor, size: Tuple[int, ...]) -> "Tensor":
        ...

    @overload
    def view(self: T_Tensor, size0: int) -> "Tensor1D":
        ...

    @overload
    def view(self: T_Tensor, size0: int, size1: int) -> "Tensor2D":
        ...

    @overload
    def view(self: T_Tensor, size0: int, size1: int, size2: int) -> "Tensor3D":
        ...

    @overload
    def view(self, dtype: torch.dtype) -> "Tensor":
        ...

    ndim: T_NDim  # type: ignore

    __eq__ = torch.Tensor.__eq__  # noqa: F811
    __getitem__ = torch.Tensor.__getitem__  # noqa: F811
    __ne__ = torch.Tensor.__ne__  # noqa: F811
    abs = torch.Tensor.abs  # noqa: F811
    contiguous = torch.Tensor.contiguous  # noqa: F811
    is_complex = torch.Tensor.is_complex  # noqa: F811
    is_floating_point = torch.Tensor.is_floating_point  # noqa: F811
    is_signed = torch.Tensor.is_signed  # noqa: F811
    item = torch.Tensor.item  # noqa: F811  # type: ignore
    mean = torch.Tensor.mean  # noqa: F811
    reshape = torch.Tensor.reshape  # noqa: F811
    squeeze = torch.Tensor.squeeze  # noqa: F811
    sum = torch.Tensor.sum  # noqa: F811
    to = torch.Tensor.to  # noqa: F811
    tolist = torch.Tensor.tolist  # noqa: F811
    unsqueeze = torch.Tensor.unsqueeze  # noqa: F811
    view = torch.Tensor.view  # noqa: F811


class Tensor(
    _TensorNDBase[Literal[None], int, BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], int, BuiltinNumber, bool, bool, bool],
):
    ...


class Tensor0D(
    _TensorNDBase[Literal[None], Literal[0], BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], Literal[0], BuiltinNumber, bool, bool, bool],
):
    ...


class Tensor1D(
    _TensorNDBase[Literal[None], Literal[1], BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], Literal[1], BuiltinNumber, bool, bool, bool],
):
    ...


class Tensor2D(
    _TensorNDBase[Literal[None], Literal[2], BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], Literal[2], BuiltinNumber, bool, bool, bool],
):
    ...


class Tensor3D(
    _TensorNDBase[Literal[None], Literal[3], BuiltinNumber, bool, bool, bool],
    metaclass=_TensorNDMeta[Literal[None], Literal[3], BuiltinNumber, bool, bool, bool],
):
    ...


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
):
    ...


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
):
    ...


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
        Literal[DTypeEnum.int8],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.int8],
        int,
        int,
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
):
    ...


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
):
    ...


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
):
    ...


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
):
    ...


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
):
    ...


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
):
    ...


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
):
    ...


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


class CHalfTensor(
    _TensorNDBase[
        Literal[DTypeEnum.chalf],
        int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.chalf],
        int,
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
        int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[DTypeEnum.cdouble],
        int,
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


class ComplexFloatingTensor(
    _TensorNDBase[
        Literal[None],
        int,
        complex,
        Literal[False],
        Literal[True],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        int,
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
        int,
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        int,
        float,
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
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[0],
        float,
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
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[1],
        float,
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
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[2],
        float,
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
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[3],
        float,
        Literal[True],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.float32


class SignedIntegerTensor(
    _TensorNDBase[
        Literal[None],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        int,
        int,
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
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[0],
        int,
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
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[1],
        int,
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
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[2],
        int,
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
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[True],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.int32


class UnsignedIntegerTensor(
    _TensorNDBase[
        Literal[None],
        int,
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        int,
        int,
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
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[0],
        int,
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
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[1],
        int,
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
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[2],
        int,
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
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
    metaclass=_TensorNDMeta[
        Literal[None],
        Literal[3],
        int,
        Literal[False],
        Literal[False],
        Literal[False],
    ],
):
    _DEFAULT_DTYPE = DTypeEnum.uint8
