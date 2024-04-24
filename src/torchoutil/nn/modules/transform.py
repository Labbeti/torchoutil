#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Tensor, nn

from torchoutil.nn.functional.transform import repeat_interleave_nd, resample_nearest
from torchoutil.utils.collections import dump_dict


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
    def __init__(self, repeats: int, dim: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return repeat_interleave_nd(x, self.repeats, self.dim)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats, dim=self.dim))


class ResampleNearest(nn.Module):
    def __init__(
        self,
        rates: Union[float, Iterable[float]],
        dims: Union[int, Iterable[int]] = -1,
    ) -> None:
        super().__init__()
        self.rates = rates
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest(x, self.rates, self.dims)

    def extra_repr(self) -> str:
        return dump_dict(dict(rates=self.rates, dims=self.dims))
