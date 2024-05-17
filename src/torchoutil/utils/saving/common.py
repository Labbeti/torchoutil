#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import Namespace
from collections import Counter
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, TypeVar, Union, overload

from torch import Tensor
from torch.types import Number as TorchNumber

from torchoutil.utils.packaging import _NUMPY_AVAILABLE, _OMEGACONF_AVAILABLE
from torchoutil.utils.type_checks import is_dataclass_instance, is_namedtuple_instance

if _NUMPY_AVAILABLE:
    import numpy as np

if _OMEGACONF_AVAILABLE:
    from omegaconf import DictConfig, ListConfig, OmegaConf


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@overload
def to_builtin(x: Enum) -> str:
    ...


@overload
def to_builtin(x: Path) -> str:
    ...


@overload
def to_builtin(x: Namespace) -> Dict[str, Any]:
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
    # Terminal cases
    if isinstance(x, (int, float, bool, complex, str, bytes)):
        return x
    elif isinstance(x, Enum):
        return x.name
    elif isinstance(x, Path):
        return str(x)
    elif isinstance(x, Tensor):
        return x.tolist()
    elif _NUMPY_AVAILABLE and isinstance(x, np.ndarray):
        return x.tolist()
    elif _NUMPY_AVAILABLE and isinstance(x, np.generic):
        return x.item()
    # Non-terminal cases
    elif _OMEGACONF_AVAILABLE and isinstance(x, (DictConfig, ListConfig)):
        return to_builtin(OmegaConf.to_container(x, resolve=False, enum_to_str=True))
    elif isinstance(x, Namespace):
        return to_builtin(x.__dict__)
    elif isinstance(x, Counter):
        return to_builtin(dict(x))
    elif is_dataclass_instance(x):
        return to_builtin(asdict(x))
    elif is_namedtuple_instance(x):
        return to_builtin(x._asdict())
    elif isinstance(x, Mapping):
        return {to_builtin(k): to_builtin(v) for k, v in x.items()}  # type: ignore
    elif isinstance(x, Iterable):
        return [to_builtin(xi) for xi in x]  # type: ignore
    else:
        return x
