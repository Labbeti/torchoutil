#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, overload

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

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


def to_safetensors(
    tensors: Dict[str, Tensor],
    fpath: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    if not is_dict_str_tensor(tensors):
        msg = f"Invalid argument type {type(tensors)}."
        raise TypeError(msg)

    return save_file(tensors, fpath, metadata)
