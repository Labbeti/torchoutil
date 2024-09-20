#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from dataclasses import asdict, dataclass
from typing import Callable, Final, Iterable, Optional, TypedDict, Union

import torch
from typing_extensions import NotRequired

from torchoutil.pyoutil.functools import identity
from torchoutil.pyoutil.importlib import package_is_available

_EXTRAS_PACKAGES = (
    "colorlog",
    "h5py",
    "numpy",
    "omegaconf",
    "pandas",
    "tensorboard",
    "tqdm",
    "yaml",
)
_EXTRA_AVAILABLE = {name: package_is_available(name) for name in _EXTRAS_PACKAGES}


_COLORLOG_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["colorlog"]
_H5PY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["h5py"]
_NUMPY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["numpy"]
_OMEGACONF_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["omegaconf"]
_PANDAS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["pandas"]
_TENSORBOARD_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tensorboard"]
_TQDM_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tqdm"]
_YAML_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["yaml"]

_VERSION_PATTERN = r"^(?P<major>\d+)\.(?P<minor>\d+)(\.(?P<micro>\d+)(.*)|)$"
_VERSION_FORMAT = r"{major}.{minor}.{micro}"


class VersionDict(TypedDict):
    major: int
    minor: int
    micro: NotRequired[Optional[int]] = None


@dataclass
class Version:
    major: int
    minor: int
    micro: Optional[int] = None

    @classmethod
    def from_dict(cls, version_dict: VersionDict) -> "Version":
        return Version(**version_dict)

    @classmethod
    def from_str(cls, version_str: str) -> "Version":
        version_dict = re.match(_VERSION_PATTERN, version_str)
        if version_dict is None:
            version_dict = {"major": 0, "minor": 0}
        else:
            version_dict = version_dict.groupdict()
            version_dict = {
                k: int(v) if v is not None else v for k, v in version_dict.items()
            }
        return Version(**version_dict)

    def to_dict(self) -> VersionDict:
        return asdict(self)

    def to_str(self) -> str:
        version_str = _VERSION_FORMAT.format(**asdict(self))
        version_str = version_str.replace(".None", "")
        return version_str


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
    version_info = Version.from_str(version)
    return version_info.major > 1 or (
        version_info.major == 1 and version_info.minor >= 13
    )
