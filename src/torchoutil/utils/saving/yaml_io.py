#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import Namespace
from pathlib import Path
from typing import Any, Literal, Mapping, Type, Union

from yaml import (
    BaseLoader,
    CBaseLoader,
    CFullLoader,
    CLoader,
    CSafeLoader,
    CUnsafeLoader,
    FullLoader,
    Loader,
    MappingNode,
    Node,
    SafeLoader,
    ScalarNode,
    SequenceNode,
    UnsafeLoader,
)
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from torchoutil.utils.packaging import _OMEGACONF_AVAILABLE, _YAML_AVAILABLE
from torchoutil.utils.saving.common import to_builtin
from torchoutil.utils.stdlib.typing import DataclassInstance, NamedTupleInstance

if _YAML_AVAILABLE:
    import yaml
else:
    raise ImportError(
        f"Cannot use python module {__file__} since pyyaml package is not installed."
    )

if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf


Loaders = Union[
    Type[Loader],
    Type[BaseLoader],
    Type[FullLoader],
    Type[SafeLoader],
    Type[UnsafeLoader],
    Type[CLoader],
    Type[CBaseLoader],
    Type[CFullLoader],
    Type[CSafeLoader],
    Type[CUnsafeLoader],
]


def to_yaml(
    data: Union[Mapping[str, Any], Namespace, DataclassInstance, NamedTupleInstance],
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    to_builtins: bool = False,
    make_parents: bool = True,
    resolve: bool = False,
    sort_keys: bool = False,
    indent: Union[int, None] = None,
    **yaml_dump_kwargs,
) -> str:
    """Dump content to yaml format."""
    if not _OMEGACONF_AVAILABLE and resolve:
        raise ValueError(
            "Cannot resolve yaml config without omegaconf package."
            "Please use resolve=False or install omegaconf with 'pip install torchoutil[extras]'."
        )

    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            fpath.parent.mkdir(parents=True, exist_ok=True)

    if resolve:
        data = OmegaConf.create(data)  # type: ignore
        data = OmegaConf.to_container(data, resolve=True)  # type: ignore

    if to_builtins:
        data = to_builtin(data)

    content = yaml.dump(data, sort_keys=sort_keys, indent=indent, **yaml_dump_kwargs)
    if fpath is not None:
        fpath.write_text(content, encoding="utf-8")
    return content


def load_yaml(
    fpath: Union[str, Path],
    *,
    Loader: Loaders = SafeLoader,
    on_error: Literal["raise", "ignore"] = "raise",
) -> Any:
    """Load content from yaml filepath."""
    try:
        with open(fpath, "r") as file:
            data = yaml.load(file, Loader=Loader)
    except (ScannerError, ParserError) as err:
        if on_error == "ignore":
            return None
        else:
            raise err
    return data


class IgnoreTagLoader(SafeLoader):
    """SafeLoader that ignores yaml tags.

    Usage:

    ```python
    >>> content = yaml.load(stream, Loader=IgnoreTagLoader)
    ```
    """

    def construct_with_tag(self, tag: str, node: Node) -> Any:
        if isinstance(node, MappingNode):
            return self.construct_mapping(node)
        elif isinstance(node, ScalarNode):
            return self.construct_scalar(node)
        elif isinstance(node, SequenceNode):
            return self.construct_sequence(node)
        else:
            raise NotImplementedError(f"Unsupported node type {type(node)}.")


IgnoreTagLoader.add_multi_constructor("!", IgnoreTagLoader.construct_with_tag)
IgnoreTagLoader.add_multi_constructor("tag:", IgnoreTagLoader.construct_with_tag)
