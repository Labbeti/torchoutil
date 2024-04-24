#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Iterable, Mapping

from torch import Tensor

from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np


def to_builtin(x: Any) -> Any:
    """Helper function to sanitize data before saving to YAML or CSV file."""
    if isinstance(x, (int, float, bool, str, bytes, complex)):
        return x
    elif isinstance(x, Path):
        return str(x)
    elif isinstance(x, Tensor):
        return x.tolist()
    elif _NUMPY_AVAILABLE and isinstance(x, np.ndarray):
        return x.tolist()
    elif _NUMPY_AVAILABLE and isinstance(x, np.generic):
        return x.item()
    elif isinstance(x, Mapping):
        return {to_builtin(k): to_builtin(v) for k, v in x.items()}  # type: ignore
    elif isinstance(x, Iterable):
        return [to_builtin(xi) for xi in x]  # type: ignore
    else:
        return x
