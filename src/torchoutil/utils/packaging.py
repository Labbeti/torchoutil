#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib.util import find_spec


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


_NUMPY_AVAILABLE = _package_is_available("numpy")
_TENSORBOARD_AVAILABLE = _package_is_available("tensorboard")
_H5PY_AVAILABLE = _package_is_available("h5py")
_OMEGACONF_AVAILABLE = _package_is_available("omegaconf")
_TQDM_AVAILABLE = _package_is_available("tqdm")
_YAML_AVAILABLE = _package_is_available("yaml")
_PANDAS_AVAILABLE = _package_is_available("pandas")
_COLORLOG_AVAILABLE = _package_is_available("colorlog")
