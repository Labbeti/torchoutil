#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Generator, Tensor, nn

from pyoutil.collections import dump_dict
from torchoutil.nn.functional.crop import CropAlign, crop_dim, crop_dims


class CropDim(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.crop.crop_dim`.
    """

    def __init__(
        self,
        target_length: int,
        align: CropAlign = "left",
        dim: int = -1,
        generator: Union[int, Generator, None] = None,
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.align: CropAlign = align
        self.dim = dim
        self.generator = generator

    def forward(self, x: Tensor) -> Tensor:
        return crop_dim(
            x,
            self.target_length,
            align=self.align,
            dim=self.dim,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                target_length=self.target_length,
                align=self.align,
                dim=self.dim,
            )
        )


class CropDims(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.crop.crop_dims`.
    """

    def __init__(
        self,
        target_lengths: Iterable[int],
        aligns: Union[CropAlign, Iterable[CropAlign]] = "left",
        dims: Iterable[int] = (-1,),
        generator: Union[int, Generator, None] = None,
    ) -> None:
        super().__init__()
        self.target_lengths = target_lengths
        self.aligns: Union[CropAlign, Iterable[CropAlign]] = aligns
        self.dims = dims
        self.generator = generator

    def forward(self, x: Tensor) -> Tensor:
        return crop_dims(
            x,
            self.target_lengths,
            aligns=self.aligns,
            dims=self.dims,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                target_lengths=self.target_lengths,
                aligns=self.aligns,
                dims=self.dims,
            )
        )
