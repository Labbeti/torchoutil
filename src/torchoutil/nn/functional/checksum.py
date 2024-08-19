#!/usr/bin/env python
# -*- coding: utf-8 -*-

import struct
import zlib
from typing import Any, Iterable, Mapping, Union

import torch
from torch import Tensor, nn

from torchoutil.types import np
from torchoutil.utils.packaging import _NUMPY_AVAILABLE


# Recursive functions
def checksum_any(x: Any, **kwargs) -> int:
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
    elif isinstance(x, np.ndarray):
        return checksum_ndarray(x, **kwargs)
    elif isinstance(x, Mapping):
        return checksum_mapping(x, **kwargs)
    elif isinstance(x, Iterable):
        return checksum_iterable(x, **kwargs)
    else:
        raise TypeError(f"Unsupported type {type(x)}.")


def checksum_iterable(x: Iterable[Any], **kwargs) -> int:
    accumulator = kwargs.pop("accumulator", 0)
    return (
        sum(
            checksum_any(xi, accumulator=accumulator + (i + 1), **kwargs) * (i + 1)
            for i, xi in enumerate(x)
        )
        + accumulator
    )


def checksum_mapping(x: Mapping[Any, Any], **kwargs) -> int:
    return checksum_iterable(x.items(), **kwargs)


def checksum_module(x: nn.Module, *, only_trainable: bool = False, **kwargs) -> int:
    """Compute a simple checksum over module parameters."""
    kwargs["only_trainable"] = only_trainable
    return checksum_iterable(
        (p for p in x.parameters() if not only_trainable or p.requires_grad),
        **kwargs,
    )


# Intermediate functions
def checksum_tensor(x: Tensor, **kwargs) -> int:
    """Compute a simple checksum of a tensor. Order of values matter for the checksum."""
    if x.ndim > 0:
        x = x.detach().flatten().cpu()

        if x.dtype == torch.bool:
            dtype = torch.int
        elif x.is_complex():
            dtype = x.real.dtype
        else:
            dtype = x.dtype

        x = x * torch.arange(1, len(x) + 1, device=x.device, dtype=dtype)
        x = x.nansum()

    x = x.item()
    x = checksum_number(x, **kwargs)
    return x


def checksum_ndarray(x: np.ndarray, **kwargs) -> int:
    if _NUMPY_AVAILABLE:
        return kwargs.get("accumulator", 0)
    else:
        return checksum_tensor(torch.from_numpy(x), **kwargs)


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
    x = struct.pack("!f", x)
    x = struct.unpack("!i", x)[0]
    return x + kwargs.get("accumulator", 0)


def checksum_int(x: int, **kwargs) -> int:
    return x + kwargs.get("accumulator", 0)


def checksum_none(x: None, **kwargs) -> int:
    return kwargs.get("accumulator", 0)
