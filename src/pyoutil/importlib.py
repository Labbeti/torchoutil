#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import inspect
from importlib.util import find_spec
from types import ModuleType
from typing import Iterable

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


def package_is_available(package_name: str) -> bool:
    """Returns True if package is installed in the current python environment."""
    try:
        return find_spec(package_name) is not None
    except AttributeError:
        # Old support for Python <= 3.6
        return False
    except (ImportError, ModuleNotFoundError):
        # Python >= 3.7
        return False


def reimport_modules(
    skipped: Iterable[str] = DEFAULT_SKIPPED,
    verbose: int = 0,
) -> None:
    """Re-import modules and functions in the caller context. This function does not work with builtins constants values."""
    skipped = dict.fromkeys(skipped)

    importlib.invalidate_caches()
    caller_globals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
    if verbose >= 1:
        print(f"{caller_globals.keys()=}")

    for k, v in caller_globals.items():
        if k in skipped:
            continue

        if isinstance(v, ModuleType):
            importlib.reload(v)
            continue

        v = inspect.getmodule(v)
        if v is None or v.__name__ == "__main__":
            continue

        importlib.reload(v)  # type: ignore
        try:
            caller_globals[k] = getattr(v, k)
        except AttributeError:
            if verbose >= 1:
                print(f"Cannot set parent global value '{k}'.")
