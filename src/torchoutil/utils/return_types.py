#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type used for backward compatibility."""

import sys
from typing import Generic, NamedTuple, Sequence

from torch import LongTensor, Tensor, __version__
from torch.torch_version import TorchVersion
from typing_extensions import TypeVar

T = TypeVar("T")


if __version__ < TorchVersion("2.0.0"):

    class namedtuple_values_indices(NamedTuple):
        values: Tensor
        indices: LongTensor

        def __init__(self, args: Sequence[Tensor]):
            super().__init__(*args)

    class min(namedtuple_values_indices):
        ...

    class max(namedtuple_values_indices):
        ...

    class topk(namedtuple_values_indices):
        ...

else:
    from torch.return_types import max, min, topk  # type: ignore # noqa: F401


if sys.version_info.major == 3 and sys.version_info.minor <= 8:
    # workaround for typing in python 3.8
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
