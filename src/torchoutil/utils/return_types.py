#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type used for backward compatibility."""

from typing import Generic, NamedTuple

from torch import Tensor, __version__
from typing_extensions import TypeVar

from torchoutil.pyoutil.semver import Version

T = TypeVar("T")
T_Values = TypeVar("T_Values", bound=Tensor, covariant=True)
T_Indices = TypeVar("T_Indices", bound=Tensor, covariant=True)


__all__ = [
    "max",
    "min",
    "sort",
    "topk",
    "shape",
    "ndim",
    "top_p",
]


class _namedtuple_values_indices(Generic[T_Values, T_Indices], tuple):
    @property
    def values(self) -> T_Values:
        return self[0]

    @property
    def indices(self) -> T_Indices:
        return self[1]


if Version(str(__version__)) < Version("2.0.0"):

    class max(_namedtuple_values_indices[Tensor, Tensor]):
        ...

    class min(_namedtuple_values_indices[Tensor, Tensor]):
        ...

    class sort(_namedtuple_values_indices[Tensor, Tensor]):
        ...

    class topk(_namedtuple_values_indices[Tensor, Tensor]):
        ...

else:
    from torch.return_types import max, min, sort, topk  # type: ignore # noqa: F401


if Version.python() < (3, 11, 0):
    # workaround for typing in python 3.8-3.10
    class _shape_base(NamedTuple):
        valid: bool
        shape: T  # type: ignore

    class shape(_shape_base, Generic[T]):  # type: ignore
        ...

else:

    class shape(NamedTuple, Generic[T]):
        valid: bool
        shape: T


class ndim(NamedTuple):
    valid: bool
    ndim: int


class top_p(_namedtuple_values_indices):
    ...
