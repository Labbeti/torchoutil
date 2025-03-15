#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import json
import logging
import sys
from importlib.metadata import Distribution, PackageNotFoundError
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Dict, List

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


def is_available_package(package: str) -> bool:
    """Returns True if package is installed in the current python environment."""
    try:
        return find_spec(package) is not None
    except AttributeError:
        # Old support for Python <= 3.6
        return False
    except (ImportError, ModuleNotFoundError):
        # Python >= 3.7
        return False


def is_editable_package(package: str) -> bool:
    # TODO: check if this works with package containing - or _
    try:
        direct_url = Distribution.from_name(package).read_text("direct_url.json")
    except PackageNotFoundError:
        return False
    if direct_url is None:
        return False
    editable = json.loads(direct_url).get("dir_info", {}).get("editable", False)
    return editable


def search_submodules(
    root: ModuleType,
    only_editable: bool = True,
    only_loaded: bool = False,
) -> List[ModuleType]:
    """Return the submodules already imported."""

    def _impl(
        root: ModuleType,
        accumulator: dict[ModuleType, None],
    ) -> dict[ModuleType, None]:
        attrs = [getattr(root, attr_name) for attr_name in dir(root)]
        submodules = [
            attr
            for attr in attrs
            if isinstance(attr, ModuleType) and attr not in accumulator
        ]
        submodules = {
            submodule
            for submodule in submodules
            if (
                (
                    not only_editable
                    or is_editable_package(submodule.__name__.split(".")[0])
                )
                and (not only_loaded or submodule.__name__ in sys.modules)
            )
        }
        accumulator.update(submodules)
        for submodule in submodules:
            accumulator = _impl(submodule, accumulator)
        return accumulator

    submodules = _impl(root, {root})
    submodules = list(submodules)
    submodules = submodules[::-1]
    return submodules


def reload_submodules(
    module: ModuleType,
    *others: ModuleType,
    verbose: int = 0,
    only_editable: bool = True,
    only_loaded: bool = False,
) -> List[ModuleType]:
    modules = (module,) + others
    candidates: Dict[ModuleType, None] = {}
    for module in modules:
        submodules = search_submodules(
            module, only_editable=only_editable, only_loaded=only_loaded
        )
        candidates.update(dict.fromkeys(submodules))

    for candidate in candidates:
        if verbose > 0:
            pylog.info(f"Reload '{candidate}'...")
        try:
            importlib.reload(candidate)
        except ModuleNotFoundError as err:
            err.add_note(
                f"Did the module '{candidate.__name__}' has been renamed after starting execution?"
            )
            raise err

    return candidates


def reload_editable_packages(*, verbose: int = 0) -> List[ModuleType]:
    pkg_names = {name.split(".")[0] for name in sys.modules.keys()}
    editable_packages = [
        sys.modules[name] for name in pkg_names if is_editable_package(name)
    ]
    if verbose >= 2:
        msg = f"{len(editable_packages)}/{len(pkg_names)} editable packages found: {editable_packages}"
        pylog.debug(msg)

    return reload_submodules(
        *editable_packages,
        verbose=verbose,
        only_editable=True,
        only_loaded=False,
    )


class Placeholder:
    """Placeholder object. All instances attributes always returns the object itself."""

    def __init__(self, *args, **kwargs) -> None:
        ...

    def __getattr__(self, name: str) -> Any:
        return self

    def __call__(self, *args, **kwargs) -> Any:
        return self

    def __getitem__(self, *args, **kwargs) -> Any:
        return self


# Aliases
package_is_available = is_available_package
