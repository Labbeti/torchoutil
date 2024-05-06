#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, TypeVar, Union, overload

from torch import Tensor
from torch.types import Number as TorchNumber

from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@overload
def to_builtin(x: Path) -> str:
    ...


@overload
def to_builtin(x: Tensor) -> Union[List, TorchNumber]:
    ...


@overload
def to_builtin(x: Mapping[K, V]) -> Dict[K, V]:
    ...


@overload
def to_builtin(x: T) -> T:
    ...


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
