#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Generator, Tensor, nn

from torchoutil.nn.functional.crop import crop_dim, crop_dims
from torchoutil.nn.functional.others import default_extra_repr


class CropDim(nn.Module):
    def __init__(
        self,
        target_length: int,
        align: str = "left",
        dim: int = -1,
        generator: Union[int, Generator, None] = None,
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.align = align
        self.dim = dim
        self.generator = generator

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return crop_dim(x, self.target_length, self.align, self.dim, self.generator)

    def extra_repr(self) -> str:
        return default_extra_repr(
            target_length=self.target_length,
            align=self.align,
            dim=self.dim,
        )


class CropDims(nn.Module):
    def __init__(
        self,
        target_lengths: Iterable[int],
        aligns: Iterable[str] = ("left",),
        dims: Iterable[int] = (-1,),
        generator: Union[int, Generator, None] = None,
    ) -> None:
        super().__init__()
        self.target_lengths = target_lengths
        self.aligns = aligns
        self.dims = dims
        self.generator = generator

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return crop_dims(x, self.target_lengths, self.aligns, self.dims, self.generator)

    def extra_repr(self) -> str:
        return default_extra_repr(
            target_lengths=self.target_lengths,
            aligns=self.aligns,
            dims=self.dims,
        )
