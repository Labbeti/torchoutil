#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Literal, Sequence, Union, overload

import torch

from torchoutil.nn.functional.get import get_device, get_dtype
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
    DeviceLike,
    DTypeLike,
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
    "as_tensor",
    "zeros",
    "ones",
    "full",
    "empty",
    "rand",
]


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
    dtype = get_dtype(dtype)
    device = get_device(device)
    return torch.as_tensor(data, dtype=dtype, device=device)


def zeros(
    *data: Any,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = get_dtype(dtype)
    device = get_device(device)
    return torch.zeros(
        *data,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


def ones(
    data: Sequence[int],
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = get_dtype(dtype)
    device = get_device(device)
    return torch.ones(
        data,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


def empty(
    data: Sequence[int],
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = get_dtype(dtype)
    device = get_device(device)
    return torch.empty(
        data,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


def rand(
    *data: Any,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = get_dtype(dtype)
    device = get_device(device)
    return torch.rand(
        *data,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


def full(
    data: Sequence[int],
    fill_value: BuiltinNumber,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    out: Union[torch.Tensor, None] = None,
    layout: Union[torch.layout, None] = None,
    pin_memory: Union[bool, None] = False,
    requires_grad: Union[bool, None] = False,
) -> torch.Tensor:
    dtype = get_dtype(dtype)
    device = get_device(device)
    return torch.full(
        data,
        fill_value=fill_value,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
