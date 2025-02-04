#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pathlib import Path
from typing import Any, Optional, Union, BinaryIO

import torch

from torchoutil.core.packaging import _TORCHAUDIO_AVAILABLE
from torchoutil.pyoutil.io import _setup_path
from torchoutil.pyoutil.typing import NoneType

if not _TORCHAUDIO_AVAILABLE:
    msg = f"Cannot use python module {__file__} since torchaudio package is not installed."
    raise ImportError(msg)

import torchaudio
from torchaudio.io import CodecConfig


def to_torchaudio(
    uri: Union[BinaryIO, str, os.PathLike],
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
    compression: Optional[Union[CodecConfig, float, int]] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
) -> bytes:
    if isinstance(f, (str, Path, os.PathLike, NoneType)):
        f = _setup_path(f, overwrite, make_parents)
        buffer = io.BytesIO()
    else:
        buffer = f

    torchaudio.save(
        src,
        uri,
        sample_rate,
        channels_first,
        format,
        encoding,
        bits_per_sample,
        buffer_size,
        backend,
        compression,
    )

    content = buffer.getvalue()
    if isinstance(f, Path):
        f.write_bytes(content)

    return content


def load_torchaudio(
    uri: Union[BinaryIO, str, os.PathLike],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
) -> Any:
    return torchaudio.load(
        uri,
        frame_offset,
        num_frames,
        normalize,
        channels_first,
        format,
        buffer_size,
        backend,
    )
