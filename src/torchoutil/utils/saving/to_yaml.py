#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Union

from torchoutil.utils.packaging import _OMEGACONF_AVAILABLE, _YAML_AVAILABLE
from torchoutil.utils.saving.common import to_builtin
from torchoutil.utils.type_checks import DataclassInstance, NamedTupleInstance

if _YAML_AVAILABLE:
    import yaml
else:
    raise ImportError(
        "Cannot use to_yaml python module since pyyaml package is not installed."
    )

if _OMEGACONF_AVAILABLE:
    from omegaconf import DictConfig, OmegaConf


def save_to_yaml(
    data: Mapping[str, Any] | Namespace | DataclassInstance | NamedTupleInstance,
    fpath: Union[str, Path, None],
    *,
    overwrite: bool = True,
    to_builtins: bool = True,
    resolve: bool = False,
    sort_keys: bool = False,
    indent: int | None = None,
    make_parents: bool = True,
    **kwargs,
) -> str:
    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            os.makedirs(fpath.parent, exist_ok=True)

    if isinstance(data, Namespace):
        data = data.__dict__

    elif is_dataclass(data) or isinstance(data, DataclassInstance):
        if isinstance(data, type):
            raise TypeError(f"Invalid argument type {type(data)}.")
        data = asdict(data)

    elif isinstance(data, NamedTupleInstance):
        data = data._asdict()

    elif _OMEGACONF_AVAILABLE and isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=resolve)  # type: ignore

    if to_builtins:
        data = to_builtin(data)

    if resolve:
        data_cfg = OmegaConf.create(data)  # type: ignore
        data = OmegaConf.to_container(data_cfg, resolve=True)  # type: ignore

    content = yaml.dump(data, sort_keys=sort_keys, indent=indent, **kwargs)
    if fpath is not None:
        fpath.write_text(content, encoding="utf-8")
    return content
