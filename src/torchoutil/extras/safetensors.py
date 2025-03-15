#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, overload

from safetensors import safe_open
from safetensors.torch import save
from torch import Tensor

from torchoutil.nn.functional.transform import as_tensor
from torchoutil.pyoutil.inspect import get_fullname
from torchoutil.pyoutil.io import _setup_path
from torchoutil.types.guards import is_dict_str_tensor


@overload
def load_safetensors(
    fpath: Union[str, Path],
    *,
    device: str = "cpu",
    return_metadata: Literal[False] = False,
) -> Dict[str, Tensor]:
    ...


@overload
def load_safetensors(
    fpath: Union[str, Path],
    *,
    device: str = "cpu",
    return_metadata: Literal[True],
) -> Tuple[Dict[str, Tensor], Dict[str, str]]:
    ...


def load_safetensors(
    fpath: Union[str, Path],
    *,
    device: str = "cpu",
    return_metadata: bool = False,
) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, str]]]:
    tensors = {}
    with safe_open(fpath, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

        if return_metadata:
            metadata = f.metadata()
            result = tensors, metadata
        else:
            result = tensors

    return result


@overload
def dump_safetensors(
    tensors: Dict[str, Tensor],
    fpath: Union[str, Path, None] = None,
    metadata: Optional[Dict[str, str]] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    convert_to_tensor: Literal[False] = False,
) -> bytes:
    ...


@overload
def dump_safetensors(
    tensors: Dict[str, Any],
    fpath: Union[str, Path, None] = None,
    metadata: Optional[Dict[str, str]] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    convert_to_tensor: Literal[True],
) -> bytes:
    ...


def dump_safetensors(
    tensors: Dict[str, Tensor],
    fpath: Union[str, Path, None] = None,
    metadata: Optional[Dict[str, str]] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    convert_to_tensor: bool = False,
) -> bytes:
    if convert_to_tensor:
        tensors = {k: as_tensor(v) for k, v in tensors.items()}
    elif not is_dict_str_tensor(tensors):
        msg = f"Invalid argument type {type(tensors)}. (expected dict[str, Tensor] but found {get_fullname(type(tensors))})"
        raise TypeError(msg)

    fpath = _setup_path(fpath, overwrite, make_parents)
    content = save(tensors, fpath, metadata)
    if fpath is not None:
        fpath.write_bytes(content)
    return content


to_safetensors = dump_safetensors
