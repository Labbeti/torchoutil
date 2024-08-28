#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Callable, Final, Iterable, Optional, Union

import torch

from pyoutil.functools import identity
from pyoutil.importlib import package_is_available

_EXTRAS_PACKAGES = (
    "numpy",
    "tensorboard",
    "h5py",
    "omegaconf",
    "tqdm",
    "yaml",
    "pandas",
    "colorlog",
)
_EXTRA_AVAILABLE = {name: package_is_available(name) for name in _EXTRAS_PACKAGES}


_NUMPY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["numpy"]
_TENSORBOARD_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tensorboard"]
_H5PY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["h5py"]
_OMEGACONF_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["omegaconf"]
_TQDM_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tqdm"]
_YAML_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["yaml"]
_PANDAS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["pandas"]
_COLORLOG_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["colorlog"]

_VERSION_PATTERN = r"^(?P<major>\d+)\.(?P<minor>\d+)(\.(?P<micro>\d+)(.*)|)$"


@dataclass
class VersionInfo:
    major: int
    minor: int
    micro: Optional[int] = None


def requires_packages(packages: Union[str, Iterable[str]]) -> Callable:
    if isinstance(packages, str):
        packages = [packages]
    else:
        packages = list(packages)

    missing = [pkg for pkg in packages if not package_is_available(pkg)]
    if len(missing) == 0:
        return identity

    prefix = "\n - "
    missing_str = prefix.join(missing)
    msg = (
        f"Cannot use/import objects because the following optionals dependencies are missing:"
        f"{prefix}{missing_str}\n"
        f"Please install them using `pip install torchoutil[extras]`."
    )
    raise ImportError(msg)


def torch_version_ge_1_13() -> bool:
    version = str(torch.__version__)
    version_info = re.match(_VERSION_PATTERN, version)
    if version_info is None:
        version_info = {"major": 0, "minor": 0}
    else:
        version_info = version_info.groupdict()
        version_info = {
            k: int(v) if v is not None else v for k, v in version_info.items()
        }

    version_info = VersionInfo(**version_info)
    return version_info.major > 1 or (
        version_info.major == 1 and version_info.minor >= 13
    )
