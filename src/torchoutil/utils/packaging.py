#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib.util import find_spec
from typing import Final


def _package_is_available(package_name: str) -> bool:
    """Returns True if package is installed in the current python environment."""
    try:
        return find_spec(package_name) is not None
    except AttributeError:
        # Old support for Python <= 3.6
        return False
    except (ImportError, ModuleNotFoundError):
        # Python >= 3.7
        return False


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
_EXTRA_AVAILABLE = {name: _package_is_available(name) for name in _EXTRAS_PACKAGES}


_NUMPY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["numpy"]
_TENSORBOARD_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tensorboard"]
_H5PY_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["h5py"]
_OMEGACONF_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["omegaconf"]
_TQDM_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["tqdm"]
_YAML_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["yaml"]
_PANDAS_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["pandas"]
_COLORLOG_AVAILABLE: Final[bool] = _EXTRA_AVAILABLE["colorlog"]
