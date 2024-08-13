#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import logging
import os
import os.path as osp
from pathlib import Path
from typing import List, Literal, Optional, Union

from torch.hub import download_url_to_file

from torchoutil.utils.packaging import _TQDM_AVAILABLE

if _TQDM_AVAILABLE:
    import tqdm


DEFAULT_CHUNK_SIZE = 256 * 1024**2  # 256 MiB
HASH_TYPES = ("sha256", "md5")
HashType = Literal["sha256", "md5"]

pylog = logging.getLogger(__name__)


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


def download_file(
    url: str,
    fpath: Union[str, Path],
    hash_prefix: Optional[str] = None,
    make_intermediate: bool = False,
    verbose: int = 0,
) -> Path:
    fpath = Path(fpath)
    if make_intermediate:
        dpath = fpath.parent
        dpath.mkdir(parents=True, exist_ok=True)

    download_url_to_file(
        url,
        fpath,  # type: ignore
        hash_prefix=hash_prefix,
        progress=verbose > 0,
    )
    return fpath


def safe_rmdir(
    root: Union[str, Path],
    *,
    rm_root: bool = True,
    error_on_non_empty_dir: bool = True,
    followlinks: bool = False,
    verbose: int = 0,
) -> List[str]:
    """Remove all empty sub-directories.

    Args:
        root: Root directory path.
        rm_root: If True, remove the root directory too if it is empty at the end. defaults to True.
        error_on_non_empty_dir: If True, raises a RuntimeError if a subdirectory contains at least 1 file. Otherwise it will ignore non-empty directories. defaults to True.
        followlinks: Indicates whether or not symbolic links shound be followed. defaults to False.
        verbose: Verbose level. defaults to 0.

    Returns:
        The list of directories paths deleted.
    """
    root = str(root)
    if not osp.isdir(root):
        raise FileNotFoundError(
            f"Target root directory does not exists. (with {root=})"
        )

    to_delete = set()

    walker = os.walk(root, topdown=False, followlinks=followlinks)
    if _TQDM_AVAILABLE:
        walker = tqdm.tqdm(walker, disable=verbose < 2)

    for dpath, dnames, fnames in walker:
        if not rm_root and dpath == root:
            continue
        elif len(fnames) == 0 and (
            all(osp.join(dpath, dname) in to_delete for dname in dnames)
        ):
            to_delete.add(dpath)
        elif error_on_non_empty_dir:
            raise RuntimeError(f"Cannot remove non-empty directory '{dpath}'.")
        elif verbose >= 2:
            pylog.debug(f"Ignoring non-empty directory '{dpath}'...")

    for dpath in to_delete:
        os.rmdir(dpath)

    return list(to_delete)
