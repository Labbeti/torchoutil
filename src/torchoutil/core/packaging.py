#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Final, Iterable, Union

import torch

from torchoutil.core.semver import Version
from torchoutil.pyoutil.functools import identity
from torchoutil.pyoutil.importlib import package_is_available

_EXTRAS_PACKAGES = (
    "colorlog",
    "h5py",
    "numpy",
    "omegaconf",
    "pandas",
    "safetensors",
    "scipy",
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
_SAFETENSORS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["safetensors"]
_SCIPY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["scipy"]
_TENSORBOARD_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tensorboard"]
_TQDM_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tqdm"]
_YAML_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["yaml"]


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
    version_str = str(torch.__version__)
    version = Version.from_str(version_str)
    return version >= Version("1.13.0")
