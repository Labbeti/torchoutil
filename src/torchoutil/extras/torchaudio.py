#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from io import BufferedWriter
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import torch

from torchoutil.core.packaging import _TORCHAUDIO_AVAILABLE
from torchoutil.pyoutil.importlib import Placeholder
from torchoutil.pyoutil.io import _setup_path
from torchoutil.pyoutil.warnings import deprecated_alias

if not _TORCHAUDIO_AVAILABLE:
    msg = f"Cannot use python module {__file__} since torchaudio package is not installed."
    raise ImportError(msg)

import torchaudio

try:
    from torchaudio.io import CodecConfig  # type: ignore
except (ImportError, AttributeError):

    class CodecConfig(Placeholder):
        ...


def dump_with_torchaudio(
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
    """Dump tensors to audio waveform. Requires torchaudio package installed."""
    if sample_rate <= 0:
        msg = f"Invalid argument {sample_rate=}. (expected positive value)"
        raise ValueError(msg)

    if isinstance(uri, (str, Path, os.PathLike)) or uri is None:
        uri = _setup_path(uri, overwrite, make_parents)

    buffer = io.BytesIO()
    torchaudio.save(  # type: ignore
        buffer,
        src,
        sample_rate,
        channels_first,
        format,
        encoding,
        bits_per_sample,
        buffer_size,
        backend,
        compression,  # type: ignore
    )
    content = buffer.getvalue()

    if isinstance(uri, Path):
        uri.write_bytes(content)
    elif isinstance(uri, (BinaryIO, BufferedWriter)):
        uri.write(content)
        uri.flush()

    return content


def load_with_torchaudio(
    uri: Union[BinaryIO, str, os.PathLike, Path],
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
    backend: Optional[str] = None,
) -> Any:
    return torchaudio.load(  # type: ignore
        uri,
        frame_offset,
        num_frames,
        normalize,
        channels_first,
        format,
        buffer_size,
        backend,
    )


@deprecated_alias(dump_with_torchaudio)
def to_torchaudio(*args, **kwargs):
    ...
