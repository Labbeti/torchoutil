#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

import torch
from torch import Generator, Tensor, nn
from torch.types import Device, Number

from torchoutil.nn.functional.pad import (
    PadAlign,
    PadValue,
    pad_and_stack_rec,
    pad_dim,
    pad_dims,
)
from torchoutil.utils.collections import dump_dict


class PadDim(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.pad.pad_dim`.
    """

    def __init__(
        self,
        target_length: int,
        align: PadAlign = "left",
        pad_value: PadValue = 0.0,
        dim: int = -1,
        mode: str = "constant",
        generator: Union[int, Generator, None] = None,
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.align: PadAlign = align
        self.pad_value = pad_value
        self.dim = dim
        self.mode = mode
        self.generator = generator

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return pad_dim(
            x,
            target_length=self.target_length,
            align=self.align,
            pad_value=self.pad_value,
            dim=self.dim,
            mode=self.mode,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                target_length=self.target_length,
                align=self.align,
                pad_value=self.pad_value,
                dim=self.dim,
                mode=self.mode,
            )
        )


class PadDims(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.pad.pad_dims`.
    """

    def __init__(
        self,
        target_lengths: Iterable[int],
        aligns: Iterable[PadAlign] = ("left",),
        pad_value: PadValue = 0.0,
        dims: Iterable[int] = (-1,),
        mode: str = "constant",
        generator: Union[int, Generator, None] = None,
    ) -> None:
        super().__init__()
        self.target_lengths = target_lengths
        self.aligns = aligns
        self.pad_value = pad_value
        self.dims = dims
        self.mode = mode
        self.generator = generator

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return pad_dims(
            x,
            target_lengths=self.target_lengths,
            aligns=self.aligns,
            pad_value=self.pad_value,
            dims=self.dims,
            mode=self.mode,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                target_lengths=self.target_lengths,
                aligns=self.aligns,
                pad_value=self.pad_value,
                dims=self.dims,
                mode=self.mode,
            )
        )


class PadAndStackRec(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.pad.pad_and_stack_rec`.
    """

    def __init__(
        self,
        pad_value: Number,
        *,
        device: Device = None,
        dtype: Union[None, torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.pad_value = pad_value
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        sequence: Union[Tensor, int, float, tuple, list],
    ) -> Tensor:
        return pad_and_stack_rec(
            sequence,
            self.pad_value,
            dtype=self.dtype,
            device=self.device,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                pad_value=self.pad_value,
                dtype=self.dtype,
                device=self.device,
            )
        )
