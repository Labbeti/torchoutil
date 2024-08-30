#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Optional, Union

from torch.hub import download_url_to_file

from torchoutil.pyoutil.hashlib import hash_file  # noqa: F401
from torchoutil.pyoutil.os import safe_rmdir  # noqa: F401

pylog = logging.getLogger(__name__)


def download_file(
    url: str,
    fpath: Union[str, Path],
    *,
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
