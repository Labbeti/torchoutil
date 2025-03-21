#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
from pathlib import Path

from torch.hub import get_dir

from torchoutil.pyoutil.functools import function_alias


def get_tmp_dir() -> Path:
    """Returns torchoutil temporary directory.

    Can be overriden with 'TORCHOUTIL_TMPDIR' environment variable.
    """
    default = tempfile.gettempdir()
    result = os.getenv("TORCHOUTIL_TMPDIR", default)
    result = Path(result).resolve().expanduser()
    return result


def get_cache_dir() -> Path:
    """Returns torchoutil cache directory for storing checkpoints, data and models.

    Defaults redirects to `torch.hub.get_dir()`, which is `~/.cache/torch/hub`.
    Can be overriden with 'TORCHOUTIL_CACHEDIR' environment variable.
    """
    default = get_dir()
    result = os.getenv("TORCHOUTIL_CACHEDIR", default)
    result = Path(result).resolve().expanduser()
    return result


@function_alias(get_dir)
def get_torch_cache_dir(*args, **kwargs):
    ...
