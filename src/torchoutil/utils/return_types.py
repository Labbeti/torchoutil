#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type used for backward compatibility."""

from typing import Generic, NamedTuple, TypeVar

from torch import Tensor, __version__
from torch.torch_version import TorchVersion

T = TypeVar("T")


if __version__ < TorchVersion("2.0.0"):

    class namedtuple_values_indices(NamedTuple):
        values: Tensor
        indices: Tensor

    class min(namedtuple_values_indices):
        ...

    class max(namedtuple_values_indices):
        ...

else:
    from torch.return_types import max, min  # type: ignore # noqa: F401


class shape(Generic[T], NamedTuple):
    valid: bool
    shape: T


class ndim(NamedTuple):
    valid: bool
    ndim: int
