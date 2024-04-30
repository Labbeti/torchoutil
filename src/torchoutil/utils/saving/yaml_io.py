#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Union

from torchoutil.utils.packaging import _OMEGACONF_AVAILABLE, _YAML_AVAILABLE
from torchoutil.utils.saving.common import to_builtin
from torchoutil.utils.type_checks import (
    DataclassInstance,
    NamedTupleInstance,
    is_dataclass_instance,
    is_namedtuple_instance,
)

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
    to_builtins: bool = False,
    make_parents: bool = True,
    resolve: bool = False,
    sort_keys: bool = False,
    indent: int | None = None,
    **kwargs,
) -> str:
    if resolve and not _OMEGACONF_AVAILABLE:
        raise ValueError(
            "Cannot resolve config for yaml without omegaconf package."
            "Please use resolve=False or install omegaconf with 'pip install torchoutil[extras]'."
        )

    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            os.makedirs(fpath.parent, exist_ok=True)

    if isinstance(data, Namespace):
        data = data.__dict__

    elif is_dataclass_instance(data):
        data = asdict(data)

    elif is_namedtuple_instance(data):
        data = data._asdict()

    elif _OMEGACONF_AVAILABLE and isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=False)  # type: ignore

    if to_builtins:
        data = to_builtin(data)

    if _OMEGACONF_AVAILABLE and resolve:
        data_cfg = OmegaConf.create(data)  # type: ignore
        data = OmegaConf.to_container(data_cfg, resolve=True)  # type: ignore

    content = yaml.dump(data, sort_keys=sort_keys, indent=indent, **kwargs)
    if fpath is not None:
        fpath.write_text(content, encoding="utf-8")
    return content


def load_yaml(fpath: Union[str, Path]) -> Any:
    with open(fpath, "r") as file:
        data = yaml.safe_load(file)
    return data
