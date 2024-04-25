#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Generic, Iterable, TypeVar, Union

import torch
from torch import Tensor, nn

from torchoutil.nn.functional.transform import (
    repeat_interleave_nd,
    resample_nearest,
    transform_drop,
)
from torchoutil.utils.collections import dump_dict

T = TypeVar("T")


class Repeat(nn.Module):
    def __init__(self, *repeats: int) -> None:
        super().__init__()
        self.repeats = repeats

    def forward(self, x: Tensor) -> Tensor:
        return x.repeat(self.repeats)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats))


class RepeatInterleave(nn.Module):
    def __init__(self, repeats: int, dim: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.repeat_interleave(self.repeats, self.dim)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats, dim=self.dim))


class RepeatInterleaveNd(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.resample_nearest`.
    """

    def __init__(self, repeats: int, dim: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return repeat_interleave_nd(x, self.repeats, self.dim)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats, dim=self.dim))


class ResampleNearest(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.resample_nearest`.
    """

    def __init__(
        self,
        rates: Union[float, Iterable[float]],
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        super().__init__()
        self.rates = rates
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest(
            x,
            rates=self.rates,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(rates=self.rates, dims=self.dims))


class TransformDrop(Generic[T], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.transform_drop`.
    """

    def __init__(
        self,
        transform: Callable[[T], T],
        p: float,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, x: T) -> T:
        return transform_drop(
            transform=self.transform,
            x=x,
            p=self.p,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(p=self.p))
