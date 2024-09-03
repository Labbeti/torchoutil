#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import math
import struct
import zlib
from typing import Any, Iterable, Mapping, Union

import torch
from torch import Tensor, nn

from torchoutil.pyoutil.inspect import get_fullname
from torchoutil.types import np
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

CHECKSUM_TYPES = (
    "int",
    "bool",
    "complex",
    "float",
    "bytes",
    "str",
    "NoneType",
    "torch.nn.Module",
    "torch.Tensor",
    "numpy.ndarray",
    "numpy.generic",
    "Mapping",
    "Iterable",
)


# Recursive functions
def checksum(x: Any, **kwargs) -> int:
    """Alias for `torchoutil.checksum_any`."""
    return checksum_any(x, **kwargs)


def checksum_any(x: Any, **kwargs) -> int:
    """Compute checksum of a single arbitrary python object."""
    if isinstance(x, (int, bool, complex, float)):
        return checksum_number(x, **kwargs)
    elif isinstance(x, bytes):
        return checksum_bytes(x, **kwargs)
    elif isinstance(x, str):
        return checksum_str(x, **kwargs)
    elif x is None:
        return checksum_none(x, **kwargs)
    elif isinstance(x, nn.Module):
        return checksum_module(x, **kwargs)
    elif isinstance(x, Tensor):
        return checksum_tensor(x, **kwargs)
    elif isinstance(x, (np.ndarray, np.generic)):
        return checksum_ndarray(x, **kwargs)
    elif isinstance(x, Mapping):
        return checksum_mapping(x, **kwargs)
    elif isinstance(x, Iterable):
        return checksum_iterable(x, **kwargs)
    else:
        msg = f"Invalid argument type {type(x)}. (expected one of {CHECKSUM_TYPES})"
        raise TypeError(msg)


def checksum_iterable(x: Iterable[Any], **kwargs) -> int:
    accumulator = kwargs.pop("accumulator", 0)
    accumulator += checksum_str(get_fullname(x), **kwargs)
    csum = sum(
        checksum_any(xi, accumulator=accumulator + (i + 1), **kwargs) * (i + 1)
        for i, xi in enumerate(x)
    )
    return csum + accumulator


def checksum_mapping(x: Mapping[Any, Any], **kwargs) -> int:
    return checksum_iterable(x.items(), **kwargs)


def checksum_module(
    x: nn.Module,
    *,
    only_trainable: bool = False,
    with_names: bool = False,
    buffers: bool = False,
    training: bool = False,
    **kwargs,
) -> int:
    """Compute a simple checksum over module parameters."""
    training = x.training
    x.train(training)

    if with_names:
        params_it = (
            (n, p)
            for n, p in x.named_parameters()
            if not only_trainable or p.requires_grad
        )
    else:
        params_it = (
            param
            for param in x.parameters()
            if not only_trainable or param.requires_grad
        )

    if not buffers:
        iterator = params_it
    elif with_names:
        buffers_it = (name_buffer for name_buffer in x.named_buffers())
        iterator = itertools.chain(params_it, buffers_it)
    else:
        buffers_it = (buffer for buffer in x.buffers())
        iterator = itertools.chain(params_it, buffers_it)

    csum = checksum_iterable(iterator, **kwargs)
    x.train(training)
    return csum


# Intermediate functions
@torch.inference_mode()
def checksum_tensor(x: Tensor, **kwargs) -> int:
    """Compute a simple checksum of a tensor. Order of values matter for the checksum."""
    if x.ndim > 0:
        if x.dtype == torch.bool:
            range_dtype = torch.int
        elif x.is_complex():
            range_dtype = x.real.dtype
        else:
            range_dtype = x.dtype

        if x.is_floating_point() or x.is_complex():
            nan_csum = checksum_float(math.nan, **kwargs)
            x = x.nan_to_num(nan_csum)

        x = x.flatten()
        arange = torch.arange(1, x.nelement() + 1, device=x.device, dtype=range_dtype)
        x = x + arange
        x = x * arange
        x = x.sum()

    xitem = x.item()
    csum = checksum_number(xitem, **kwargs)
    return csum


def checksum_ndarray(x: Union[np.ndarray, np.generic], **kwargs) -> int:
    if not _NUMPY_AVAILABLE:
        raise NotImplementedError(
            "Cannot call function 'checksum_ndarray' because optional dependancy 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
        )
    return checksum_tensor(torch.as_tensor(x), **kwargs)


def checksum_number(x: Union[int, bool, complex, float], **kwargs) -> int:
    """Compute a simple checksum of a builtin scalar number."""
    if isinstance(x, bool):
        return checksum_bool(x, **kwargs)
    elif isinstance(x, int):
        return checksum_int(x, **kwargs)
    elif isinstance(x, complex):
        return checksum_complex(x, **kwargs)
    elif isinstance(x, float):
        return checksum_float(x, **kwargs)
    else:
        raise TypeError(
            f"Invalid argument type {type(x)}. (expected int, bool, complex or float)"
        )


def checksum_str(x: str, **kwargs) -> int:
    return checksum_bytes(x.encode(), **kwargs)


def checksum_complex(x: complex, **kwargs) -> int:
    return checksum_tensor(torch.as_tensor([x.real, x.imag]), **kwargs)


# Terminate functions
def checksum_bool(x: bool, **kwargs) -> int:
    return int(x) + kwargs.get("accumulator", 0)


def checksum_bytes(x: bytes, **kwargs) -> int:
    return zlib.crc32(x) % (1 << 32) + kwargs.get("accumulator", 0)


def checksum_float(x: float, **kwargs) -> int:
    xint = _interpret_float_as_int(x)
    return xint + kwargs.get("accumulator", 0)


def checksum_int(x: int, **kwargs) -> int:
    return x + kwargs.get("accumulator", 0)


def checksum_none(x: None, **kwargs) -> int:
    return kwargs.get("accumulator", 0)


def _interpret_float_as_int(x: float) -> int:
    xbytes = struct.pack("!f", x)
    xint = struct.unpack("!i", xbytes)[0]
    return xint
