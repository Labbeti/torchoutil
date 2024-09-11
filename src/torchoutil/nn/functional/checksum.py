#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import math
import struct
import zlib
from dataclasses import asdict
from typing import Any, Callable, Iterable, Mapping, Union

import torch
from torch import Tensor, nn

from torchoutil.nn.functional.others import is_complex, is_floating_point, nelement
from torchoutil.pyoutil.inspect import get_fullname
from torchoutil.pyoutil.typing import (
    BuiltinNumber,
    BuiltinScalar,
    DataclassInstance,
    NamedTupleInstance,
    NoneType,
)
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
    "NamedTuple",
    "Dataclass",
    "Mapping",
    "Iterable",
)


# Recursive functions
def checksum(x: Any, **kwargs) -> int:
    """Alias for `torchoutil.checksum_any`."""
    return checksum_any(x, **kwargs)


def checksum_any(x: Any, **kwargs) -> int:
    """Compute checksum of an arbitrary python object."""
    if isinstance(x, (int, bool, complex, float)):
        return checksum_builtin_number(x, **kwargs)
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
    elif isinstance(x, NamedTupleInstance):
        return checksum_namedtuple(x, **kwargs)
    elif isinstance(x, DataclassInstance):
        return checksum_dataclass(x, **kwargs)
    elif isinstance(x, Mapping):
        return checksum_mapping(x, **kwargs)
    elif isinstance(x, Iterable):
        return checksum_iterable(x, **kwargs)
    else:
        msg = f"Invalid argument type {type(x)}. (expected one of {CHECKSUM_TYPES})"
        raise TypeError(msg)


def checksum_dataclass(x: DataclassInstance, **kwargs) -> int:
    accumulator = kwargs.pop("accumulator", 0) + checksum_str(get_fullname(x), **kwargs)
    kwargs["accumulator"] = accumulator
    return checksum_mapping(asdict(x), **kwargs)


def checksum_iterable(x: Iterable[Any], **kwargs) -> int:
    accumulator = kwargs.pop("accumulator", 0) + checksum_str(get_fullname(x), **kwargs)
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


def checksum_namedtuple(x: NamedTupleInstance, **kwargs) -> int:
    accumulator = kwargs.pop("accumulator", 0) + checksum_str(get_fullname(x), **kwargs)
    kwargs["accumulator"] = accumulator
    return checksum_mapping(x._asdict(), **kwargs)


# Intermediate functions
@torch.inference_mode()
def checksum_tensor(x: Tensor, **kwargs) -> int:
    """Compute a simple checksum of a tensor. Order of values matter for the checksum."""
    return _checksum_tensor_array_like(
        x,
        nan_to_num_fn=torch.nan_to_num,
        arange_fn=torch.arange,
    )


def checksum_ndarray(x: Union[np.ndarray, np.generic], **kwargs) -> int:
    if not _NUMPY_AVAILABLE:
        msg = "Cannot call function 'checksum_ndarray' because optional dependancy 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
        raise NotImplementedError(msg)

    # Supports non-numeric numpy arrays (byte string, unicode string, object, void)
    if x.dtype.kind in ("S", "U", "O", "V"):
        return checksum_iterable(x.tolist(), **kwargs)

    return _checksum_tensor_array_like(
        x,
        nan_to_num_fn=np.nan_to_num,
        arange_fn=np.arange,
    )


def checksum_builtin_scalar(x: BuiltinScalar, **kwargs) -> int:
    if isinstance(x, BuiltinNumber):
        return checksum_builtin_number(x, **kwargs)
    elif isinstance(x, bytes):
        return checksum_bytes(x, **kwargs)
    elif isinstance(x, NoneType):
        return checksum_none(x, **kwargs)
    elif isinstance(x, str):
        return checksum_str(x, **kwargs)
    else:
        msg = f"Invalid argument type {type(x)}. (expected int, bool, complex float, bytes, None, or str)"
        raise TypeError(msg)


def checksum_builtin_number(x: BuiltinNumber, **kwargs) -> int:
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
        msg = f"Invalid argument type {type(x)}. (expected int, bool, complex or float)"
        raise TypeError(msg)


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


def _checksum_tensor_array_like(
    x: Union[Tensor, np.ndarray, np.generic],
    *,
    nan_to_num_fn: Callable,
    arange_fn: Callable,
    **kwargs,
) -> int:
    if x.ndim == 0:
        return checksum_builtin_number(x.item(), **kwargs)

    if x.dtype == bool:
        range_dtype = int
    elif x.dtype == torch.bool:
        range_dtype = torch.int
    elif is_complex(x):
        range_dtype = x.real.dtype  # type: ignore
    else:
        range_dtype = x.dtype

    if is_floating_point(x) or is_complex(x):
        nan_csum = checksum_float(math.nan, **kwargs)
        neginf_csum = checksum_float(-math.inf, **kwargs)
        posinf_csum = checksum_float(math.inf, **kwargs)
        x = nan_to_num_fn(
            x,
            nan=nan_csum,
            neginf=neginf_csum,
            posinf=posinf_csum,
        )

    x = x.flatten()
    arange = arange_fn(1, nelement(x) + 1, dtype=range_dtype)
    x = x + arange
    x = x * arange
    x = x.sum()

    return checksum_builtin_number(x.item(), **kwargs)
