#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import urllib
import warnings
from email.message import Message
from pathlib import Path, PosixPath
from typing import Optional, Union
from urllib.error import HTTPError, URLError

from torch.hub import download_url_to_file

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
    """
    if make_intermediate is not None:
        warnings.warn(
            f"Deprecated argument {make_intermediate=}. Use make_parents={make_intermediate} instead."
        )
        make_parents = make_intermediate

    if dst is None:
        dst = "."

    dst = Path(dst)

    if dst.is_dir():
        fname = _get_filename_from_url(url)
        fpath = dst.joinpath(fname)
    elif dst.is_file():
        fpath = dst
    elif dst.exists():
        raise FileExistsError(
            f"Destination '{dst}' exists but is not a file or directory."
        )
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
        response = urllib.request.urlopen(url)
        header = response.headers.get("Content-Disposition", "")
        message = Message()
        message["content-type"] = header
        filename = message.get_param("filename", None)
    except (URLError, ValueError):
        filename = None

    if filename is None:
        filename = PosixPath(url).name.split("?")[0]
    return filename
