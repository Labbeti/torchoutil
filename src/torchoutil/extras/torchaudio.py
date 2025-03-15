#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import torch
from typing_extensions import TypeAlias

from torchoutil.core.packaging import _TORCHAUDIO_AVAILABLE
from torchoutil.pyoutil.io import _setup_path
from torchoutil.pyoutil.typing import NoneType

if not _TORCHAUDIO_AVAILABLE:
    msg = f"Cannot use python module {__file__} since torchaudio package is not installed."
    raise ImportError(msg)

import torchaudio

try:
    from torchaudio.io import CodecConfig
except ImportError:
    CodecConfig: TypeAlias = int


def load_torchaudio(
    uri: Union[BinaryIO, str, os.PathLike, Path],
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


def dump_torchaudio(
    src: torch.Tensor,
    uri: Union[BinaryIO, str, Path, os.PathLike, None],
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
    if sample_rate <= 0:
        msg = f"Invalid argument {sample_rate=}. (expected positive value)"
        raise ValueError(msg)

    buffer: Union[BinaryIO, io.BytesIO]
    if isinstance(uri, (str, Path, os.PathLike, NoneType)):
        uri = _setup_path(uri, overwrite, make_parents)
        buffer = io.BytesIO()
    else:
        buffer = uri

    torchaudio.save(
        buffer,
        src,
        sample_rate,
        channels_first,
        format,
        encoding,
        bits_per_sample,
        buffer_size,
        backend,
        compression,
    )

    if isinstance(buffer, io.BytesIO):
        content = buffer.getvalue()
    else:
        content = buffer.read()

    if isinstance(uri, Path):
        uri.write_bytes(content)

    return content


to_torchaudio = dump_torchaudio
