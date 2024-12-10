#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, Optional, Union

from safetensors.torch import load_file, save_file
from torch import Tensor

from torchoutil.types.guards import is_dict_str_tensor


def load_safetensors(
    fpath: Union[str, Path],
    *,
    device: str = "cpu",
) -> Dict[str, Tensor]:
    return load_file(fpath, device)


def to_safetensors(
    tensors: Dict[str, Tensor],
    fpath: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    if not is_dict_str_tensor(tensors):
        msg = f"Invalid argument type {type(tensors)}."
        raise TypeError(msg)

    return save_file(tensors, fpath, metadata)
