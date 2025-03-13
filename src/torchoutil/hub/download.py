#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import urllib
import warnings
from email.message import Message
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from torch.hub import download_url_to_file

# for backward compatibility
from torchoutil.pyoutil.hashlib import hash_file  # noqa: F401
from torchoutil.pyoutil.os import safe_rmdir  # noqa: F401


pylog = logging.getLogger(__name__)


def download_file(
    url: str,
    dst: Union[str, Path, None] = ".",
    *,
    hash_prefix: Optional[str] = None,
    make_parents: bool = False,
    make_intermediate: Optional[bool] = None,
    verbose: int = 0,
) -> Path:
    """Download file to target filepath or directory.

    Args:
        url: Target URL.
        dst: Target filepath or directory.
        hash_prefix: Optional hash prefix present in destination filename. defaults to None.
        make_parents: If True, make intermediate directories to destination. defaults to False.
        make_intermediate: Deprecated: alias for 'make_parents'. If not None, overwrite any value of 'make_parents'. defaults to None.
        verbose: Verbose level. defaults to 0.

    Returns:
        Path to the downloaded file.
    """
    if make_intermediate is not None:
        msg = f"Deprecated argument {make_intermediate=}. Use make_parents={make_intermediate} instead."
        warnings.warn(msg)
        make_parents = make_intermediate

    if dst is None:
        dst = "."

    dst = Path(dst)

    if dst.is_dir():
        fname = _get_filename_from_url(url)
        fpath = dst.joinpath(fname)
    elif dst.exists() and not dst.is_file():
        msg = f"Destination '{dst}' exists but is not a file or directory."
        raise FileExistsError(msg)
    else:
        fpath = dst
    del dst

    if make_parents:
        dpath = fpath.parent
        dpath.mkdir(parents=True, exist_ok=True)

    try:
        download_url_to_file(
            url,
            fpath,  # type: ignore
            hash_prefix=hash_prefix,
            progress=verbose > 0,
        )

    except HTTPError as err:
        msg = f"Cannot download from {url=}. (with {fpath=}, {hash_prefix=}, {make_intermediate=})"
        pylog.error(msg)
        raise err

    return fpath


def _get_filename_from_url(url: str) -> str:
    try:
        response = urlopen(url)
        header = response.headers.get("Content-Disposition", "")
        message = Message()
        message["content-type"] = header
        filename = message.get_param("filename", None)
    except (URLError, ValueError):
        pylog.warning(f"Cannot get target filename from {url=}. Try to detect it from URL string.")
        filename = None

    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    return str(filename)
