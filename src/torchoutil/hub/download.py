#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
from pathlib import Path
from typing import Literal, Union

DEFAULT_CHUNK_SIZE = 256 * 1024**2  # 256 MiB
HASH_TYPES = ("sha256", "md5")
HashType = Literal["sha256", "md5"]


def hash_file(
    fpath: Union[str, Path],
    hash_type: HashType,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Return the hash value for a file.

    BASED ON https://github.com/pytorch/audio/blob/v0.13.0/torchaudio/datasets/utils.py#L110
    """
    if hash_type == "sha256":
        hasher = hashlib.sha256()
    elif hash_type == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(
            f"Invalid argument hash_type={hash_type}. (expected one of {HASH_TYPES})"
        )

    with open(fpath, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    hash_value = hasher.hexdigest()
    return hash_value
