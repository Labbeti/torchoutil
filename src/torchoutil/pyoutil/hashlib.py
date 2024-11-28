#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import logging
from pathlib import Path
from typing import Literal, Protocol, Union, runtime_checkable

from typing_extensions import Buffer

HashName = Literal["sha256", "md5"]

DEFAULT_CHUNK_SIZE = 256 * 1024**2  # 256 MiB
HASH_NAMES = ("sha256", "md5")

pylog = logging.getLogger(__name__)


@runtime_checkable
class Hasher(Protocol):
    digest_size: int
    block_size: int
    name: str

    def hexdigest(self) -> str:
        ...

    def update(self, data: Buffer, /) -> None:
        ...


def hash_file(
    fpath: Union[str, Path],
    hash_type: Union[HashName, Hasher],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Return the hash value for a file.

    BASED ON https://github.com/pytorch/audio/blob/v0.13.0/torchaudio/datasets/utils.py#L110
    """
    if hash_type == "sha256":
        hasher = hashlib.sha256()
    elif hash_type == "md5":
        hasher = hashlib.md5()
    elif isinstance(hash_type, Hasher):
        hasher = hash_type
    else:
        msg = f"Invalid argument hash_type={hash_type}. (expected one of {HASH_NAMES} or a custom hasher)"
        raise ValueError(msg)
    del hash_type

    with open(fpath, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    hash_value = hasher.hexdigest()
    return hash_value
