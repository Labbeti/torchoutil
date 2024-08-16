#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type used for backward compatibility."""

from typing import NamedTuple

from torch import Tensor, __version__
from torch.torch_version import TorchVersion

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
