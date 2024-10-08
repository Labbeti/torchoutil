#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict

from .typing.classes import DataclassInstance
from .typing.guards import is_dataclass_instance  # noqa: F401


def get_defaults_values(dataclass: DataclassInstance) -> Dict[str, Any]:
    defaults = {
        f.name: f.default_factory() if callable(f.default_factory) else f.default
        for f in dataclass.__dataclass_fields__.values()
    }
    return defaults
