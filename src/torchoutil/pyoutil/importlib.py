#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import inspect
import logging
import pkgutil
import sys
from importlib.util import find_spec
from types import ModuleType
from typing import Iterable, List

DEFAULT_SKIPPED = (
    "reimport_all",
    "get_ipython",
    "exit",
    "quit",
    "__name__",
    "__doc__",
    "__package__",
    "__loader__",
    "__spec__",
    "__builtin__",
    "__builtins__",
)

pylog = logging.getLogger(__name__)


def package_is_available(package: str) -> bool:
    """Returns True if package is installed in the current python environment."""
    try:
        return find_spec(package) is not None
    except AttributeError:
        # Old support for Python <= 3.6
        return False
    except (ImportError, ModuleNotFoundError):
        # Python >= 3.7
        return False


def search_imported_submodules(
    root: ModuleType,
    parent_name: str = "",
) -> List[ModuleType]:
    """Return the submodules already imported."""
    if root.__package__ is None:
        raise ValueError(f"Cannot search in module '{root}'. (found __package__=None)")

    if parent_name != "":
        parent_name = ".".join([parent_name, root.__package__])
    else:
        parent_name = root.__package__

    if hasattr(root, "__path__"):
        paths = root.__path__
    elif root.__file__ is not None:
        paths = [root.__file__]
    else:
        raise ValueError

    module_infos = list(pkgutil.iter_modules(paths))
    candidates = []

    for info in module_infos:
        if info.name == "__main__":
            continue
        fullname = f"{parent_name}.{info.name}"
        candidate = sys.modules.get(fullname)
        if candidate is None:
            continue
        sub_candidates = search_imported_submodules(candidate, parent_name)
        candidates += sub_candidates + [candidate]

    return candidates


def reload_submodules(root: ModuleType, verbose: int = 0) -> List[ModuleType]:
    candidates = search_imported_submodules(root) + [root]
    for candidate in candidates:
        if verbose > 0:
            pylog.info(f"Reload '{candidate}'")
        importlib.reload(candidate)
    return candidates


def reload_globals_modules(
    skipped: Iterable[str] = DEFAULT_SKIPPED,
    verbose: int = 0,
) -> List[ModuleType]:
    """Re-import modules and functions in the caller context. This function does not work with builtins constants values."""
    skipped = dict.fromkeys(skipped)

    importlib.invalidate_caches()
    caller_globals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
    if verbose >= 1:
        print(f"{caller_globals.keys()=}")

    candidates = []

    for k, v in caller_globals.items():
        if k in skipped:
            continue

        if isinstance(v, ModuleType):
            candidates += reload_submodules(v, verbose=verbose)
            continue

        v = inspect.getmodule(v)
        if v is None or v.__name__ == "__main__":
            continue

        candidates += reload_submodules(v, verbose=verbose)

        try:
            caller_globals[k] = getattr(v, k)
        except AttributeError:
            if verbose >= 1:
                print(f"Cannot set parent global value '{k}'.")

    return candidates
