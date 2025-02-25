#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Literal, Optional, Sequence, Tuple, Union, overload

import torch
from torch import Tensor
from torch.types import Number
from typing_extensions import Never

from torchoutil.core.make import (
    DeviceLike,
    DTypeLike,
    GeneratorLike,
    make_device,
    make_dtype,
    make_generator,
)
from torchoutil.pyoutil.typing import BuiltinNumber
from torchoutil.types import (
    BoolTensor0D,
    BoolTensor1D,
    BoolTensor2D,
    BoolTensor3D,
    CFloatTensor0D,
    CFloatTensor1D,
    CFloatTensor2D,
    CFloatTensor3D,
    FloatTensor0D,
    FloatTensor1D,
    FloatTensor2D,
    FloatTensor3D,
    LongTensor0D,
    LongTensor1D,
    LongTensor2D,
    LongTensor3D,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
)

__all__ = [
    "arange",
    "as_tensor",
    "empty",
    "full",
    "rand",
    "randint",
    "randperm",
    "ones",
    "zeros",
]

# ----------
# arange
# ----------


@overload
def arange(
    end: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D:
    ...


@overload
def arange(
    start: Number,
    end: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D:
    ...


@overload
def arange(
    start: Number,
    end: Number,
    step: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D:
    ...


@overload
def arange(
    end: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D:
    ...


@overload
def arange(
    start: Number,
    end: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D:
    ...


@overload
def arange(
    start: Number,
    end: Number,
    step: Number,
    *,
    out: Optional[Tensor] = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D:
    ...


def arange(
    arg0: Number,
    *args: Number,
    out: Optional[Tensor] = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D:
    dtype = make_dtype(dtype)
    device = make_device(device)
    return torch.arange(
        arg0,
        *args,
        out=out,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )


# ----------
# as_tensor
# ----------


# Empty lists
@overload
def as_tensor(
    data: Sequence[Never],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> Tensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Never]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> Tensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[Never]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> Tensor3D:
    ...


# bool
@overload
def as_tensor(
    data: bool,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> BoolTensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[bool],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> BoolTensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[bool]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> BoolTensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[bool]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> BoolTensor3D:
    ...


# int
@overload
def as_tensor(
    data: int,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> LongTensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[int],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> LongTensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[int]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> LongTensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[int]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> LongTensor3D:
    ...


# float
@overload
def as_tensor(
    data: float,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> FloatTensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[float],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> FloatTensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[float]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> FloatTensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[float]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> FloatTensor3D:
    ...


# complex
@overload
def as_tensor(
    data: complex,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> CFloatTensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[complex],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> CFloatTensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[complex]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> CFloatTensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[complex]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> CFloatTensor3D:
    ...


# BuiltinNumber
@overload
def as_tensor(
    data: BuiltinNumber,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[BuiltinNumber],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[BuiltinNumber]],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[BuiltinNumber]]],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor3D:
    ...


def as_tensor(
    data: Any,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    dtype = make_dtype(dtype)
    device = make_device(device)
    return torch.as_tensor(data, dtype=dtype, device=device)


# ----------
# empty
# ----------


@overload
def empty(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D:
    ...


@overload
def empty(
    size: Tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def empty(
    size: Tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def empty(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


@overload
def empty(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def empty(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def empty(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


def empty(
    *data,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = make_dtype(dtype)
    device = make_device(device)
    return torch.empty(
        *data,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# ----------
# full
# ----------


# bool
@overload
def full(
    size: Sequence[Never],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor0D:
    ...


@overload
def full(
    size: Tuple[int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor1D:
    ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor2D:
    ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: bool,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor3D:
    ...


# int
@overload
def full(
    size: Sequence[Never],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor0D:
    ...


@overload
def full(
    size: Tuple[int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor1D:
    ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor2D:
    ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: int,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor3D:
    ...


# float
@overload
def full(
    size: Sequence[Never],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor0D:
    ...


@overload
def full(
    size: Tuple[int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor1D:
    ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor2D:
    ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: float,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor3D:
    ...


# complex
@overload
def full(
    size: Sequence[Never],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor0D:
    ...


@overload
def full(
    size: Tuple[int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor1D:
    ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor2D:
    ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: complex,
    *,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> BoolTensor3D:
    ...


# BuiltinNumber
@overload
def full(
    size: Sequence[Never],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D:
    ...


@overload
def full(
    size: Tuple[int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def full(
    size: Tuple[int, int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def full(
    size: Tuple[int, int, int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


def full(
    size: Sequence[int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = make_dtype(dtype)
    device = make_device(device)
    return torch.full(
        size,
        fill_value=fill_value,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# ----------
# ones
# ----------


@overload
def ones(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D:
    ...


@overload
def ones(
    size: Tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def ones(
    size: Tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def ones(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


@overload
def ones(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def ones(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def ones(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


def ones(
    *data,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = make_dtype(dtype)
    device = make_device(device)
    return torch.ones(
        *data,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# ----------
# rand
# ----------
@overload
def rand(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D:
    ...


@overload
def rand(
    size: Tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def rand(
    size: Tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def rand(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


@overload
def rand(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def rand(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def rand(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


def rand(
    *data: Any,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    generator: GeneratorLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = make_dtype(dtype)
    device = make_device(device)
    generator = make_generator(generator)

    return torch.rand(
        *data,
        generator=generator,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# ----------
# randint
# ----------


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[()],
    *,
    generator: GeneratorLike = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor0D:
    ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int],
    *,
    generator: GeneratorLike = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor1D:
    ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int, int],
    *,
    generator: GeneratorLike = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor2D:
    ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int, int, int],
    *,
    generator: GeneratorLike = None,
    dtype: Literal[None] = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> LongTensor3D:
    ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[()],
    *,
    generator: GeneratorLike = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor0D:
    ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int],
    *,
    generator: GeneratorLike = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor1D:
    ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int, int],
    *,
    generator: GeneratorLike = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor2D:
    ...


@overload
def randint(
    low: int,
    high: int,
    size: Tuple[int, int, int],
    *,
    generator: GeneratorLike = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor3D:
    ...


def randint(
    low: int,
    high: int,
    size: Tuple[int, ...],
    *,
    generator: GeneratorLike = None,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor:
    dtype = make_dtype(dtype)
    device = make_device(device)
    generator = make_generator(generator)
    return torch.randint(
        low=low,
        high=high,
        size=size,
        generator=generator,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )


# ----------
# randperm
# ----------


@overload
def randperm(
    n: int,
    *,
    generator: GeneratorLike = None,
    out: Optional[Tensor] = None,
    dtype: Literal[None] = None,
    layout: Optional[torch.layout] = None,
    device: DeviceLike = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
) -> LongTensor1D:
    ...


def randperm(
    n: int,
    *,
    generator: GeneratorLike = None,
    out: Optional[Tensor] = None,
    dtype: DTypeLike = None,
    layout: Optional[torch.layout] = None,
    device: DeviceLike = None,
    pin_memory: Optional[bool] = False,
    requires_grad: Optional[bool] = False,
) -> Tensor1D:
    dtype = make_dtype(dtype)
    device = make_device(device)
    generator = make_generator(generator)
    return torch.randperm(
        n=n,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# ----------
# zeros
# ----------


@overload
def zeros(
    size: Sequence[Never],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor0D:
    ...


@overload
def zeros(
    size: Tuple[int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def zeros(
    size: Tuple[int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def zeros(
    size: Tuple[int, int, int],
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


@overload
def zeros(
    size0: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor1D:
    ...


@overload
def zeros(
    size0: int,
    size1: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor2D:
    ...


@overload
def zeros(
    size0: int,
    size1: int,
    size2: int,
    /,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> Tensor3D:
    ...


def zeros(
    *data,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = make_dtype(dtype)
    device = make_device(device)
    return torch.zeros(
        *data,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
