#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import logging
from io import BufferedReader
from pathlib import Path
from typing import Literal, Optional, Protocol, Union, get_args, runtime_checkable

from typing_extensions import Buffer

HashName = Literal["sha256", "md5"]

DEFAULT_CHUNK_SIZE = 256 * 1024**2  # 256 MiB

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
    fpath: Union[str, Path, BufferedReader],
    hash_type: Union[HashName, Hasher] = "md5",
    chunk_size: Optional[int] = DEFAULT_CHUNK_SIZE,
) -> str:
    """Return the hash value for a file.

    Based on https://github.com/pytorch/audio/blob/v0.13.0/torchaudio/datasets/utils.py#L110

    Args:
        fpath: Path to existing file.
        hash_type: Hash name or custom Hasher algorithm.
        chunk_size: Max chunk size in bytes. defaults to 268435456 (256 MiB).

    Returns:
        Hash value as string.
    """
    if isinstance(fpath, (str, Path)):
        with open(fpath, "rb") as file:
            return hash_file(file, hash_type, chunk_size)
    else:
        file = fpath
    del fpath

    if hash_type == "sha256":
        hasher = hashlib.sha256()
    elif hash_type == "md5":
        hasher = hashlib.md5()
    elif isinstance(hash_type, Hasher):
        hasher = hash_type
    else:
        msg = f"Invalid argument hash_type={hash_type}. (expected one of {get_args(HashName)} or a custom hasher)"
        raise ValueError(msg)
    del hash_type

    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)

    hash_value = hasher.hexdigest()
    return hash_value
