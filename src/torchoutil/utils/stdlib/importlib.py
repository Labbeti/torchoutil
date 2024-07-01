#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import inspect
from types import ModuleType


def reimport_modules() -> None:
    """Re-import modules and functions in the caller context. This function does not work with builtins constants values."""
    importlib.invalidate_caches()

    caller_globals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
    print(f"{caller_globals.keys()=}")

    for k, v in caller_globals.items():
        if k in (
            "reimport_all",
            "__builtin__",
            "__builtins__",
            "get_ipython",
            "exit",
        ):
            ...

        elif isinstance(v, ModuleType):
            importlib.reload(v)

        else:
            v = inspect.getmodule(v)
            if v is None or v.__name__ == "__main__":
                continue

            importlib.reload(v)  # type: ignore
            try:
                caller_globals[k] = getattr(v, k)
            except AttributeError:
                pass
