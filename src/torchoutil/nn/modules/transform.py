#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Generic, Iterable, TypeVar, Union

import torch
from torch import Generator, Tensor, nn

from torchoutil.nn.functional.transform import (
    repeat_interleave_nd,
    resample_nearest_freqs,
    resample_nearest_rates,
    resample_nearest_steps,
    transform_drop,
)
from torchoutil.utils.collections import dump_dict

T = TypeVar("T")


class RepeatInterleaveNd(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.repeat_interleave_nd`.
    """

    def __init__(self, repeats: int, dim: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return repeat_interleave_nd(x, self.repeats, self.dim)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats, dim=self.dim))


class ResampleNearestRates(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.resample_nearest_rates`.
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
        return resample_nearest_rates(
            x,
            rates=self.rates,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(rates=self.rates, dims=self.dims))


class ResampleNearestFreqs(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.resample_nearest_freqs`.
    """

    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest_freqs(
            x,
            orig_freq=self.orig_freq,
            new_freq=self.new_freq,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(orig_freq=self.orig_freq, new_freq=self.new_freq, dims=self.dims)
        )


class ResampleNearestSteps(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.resample_nearest_steps`.
    """

    def __init__(
        self,
        steps: Union[float, Iterable[float]],
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest_steps(
            x,
            steps=self.steps,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(steps=self.steps, dims=self.dims))


class TransformDrop(Generic[T], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.transform_drop`.
    """

    def __init__(
        self,
        transform: Callable[[T], T],
        p: float,
        generator: Union[int, Generator, None] = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.p = p
        self.generator = generator

    def forward(self, x: T) -> T:
        return transform_drop(
            transform=self.transform,
            x=x,
            p=self.p,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(p=self.p))
