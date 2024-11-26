#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import itertools
import math
import pickle
import struct
import zlib
from dataclasses import asdict
from types import FunctionType, MethodType
from typing import Any, Callable, Iterable, Literal, Mapping, Union

import torch
from torch import Tensor, nn

from torchoutil.core.packaging import _NUMPY_AVAILABLE
from torchoutil.extras.numpy import np
from torchoutil.nn.functional.others import is_complex, is_floating_point, nelement
from torchoutil.pyoutil.inspect import get_fullname
from torchoutil.pyoutil.typing import (
    BuiltinNumber,
    BuiltinScalar,
    DataclassInstance,
    NamedTupleInstance,
    NoneType,
)

Checksumable = Union[
    int,
    bool,
    complex,
    float,
    NoneType,
    str,
    bytes,
    bytearray,
    nn.Module,
    Tensor,
    np.ndarray,
    np.generic,
    NamedTupleInstance,
    DataclassInstance,
    Mapping,
    Iterable,
    MethodType,
    FunctionType,
    functools.partial,
    type,
]
CHECKSUMABLE_TYPES = (
    "int",
    "bool",
    "complex",
    "float",
    "NoneType",
    "str",
    "bytes",
    "bytearray",
    "torch.nn.Module",
    "torch.Tensor",
    "numpy.ndarray",
    "numpy.generic",
    "NamedTuple",
    "Dataclass",
    "Mapping",
    "Iterable",
    "MethodType",
    "FunctionType",
    "functools.partial",
    "type",
)
UnkMode = Literal["pickle", "error"]
UNK_MODES = ("pickle", "error")


# Recursive functions
def checksum(
    x: Checksumable,
    *,
    unk_mode: UnkMode = "error",
    allow_protocol: bool = True,
    **kwargs,
) -> int:
    """Alias for `torchoutil.checksum_any`."""
    return checksum_any(
        x,
        unk_mode=unk_mode,
        allow_protocol=allow_protocol,
        **kwargs,
    )


def checksum_any(
    x: Checksumable,
    *,
    unk_mode: UnkMode = "error",
    allow_protocol: bool = True,
    **kwargs,
) -> int:
    """Compute checksum of an arbitrary python object."""
    kwargs.update(
        dict(
            unk_mode=unk_mode,
            allow_protocol=allow_protocol,
        )
    )
    if isinstance(x, (int, bool, complex, float)):
        return checksum_builtin_number(x, **kwargs)
    elif x is None:
        return checksum_none(x, **kwargs)
    elif isinstance(x, str):
        return checksum_str(x, **kwargs)
    elif isinstance(x, bytes):
        return checksum_bytes(x, **kwargs)
    elif isinstance(x, bytearray):
        return checksum_bytearray(x, **kwargs)
    elif isinstance(x, nn.Module):
        return checksum_module(x, **kwargs)
    elif isinstance(x, Tensor):
        return checksum_tensor(x, **kwargs)
    elif isinstance(x, (np.ndarray, np.generic)):
        return checksum_ndarray(x, **kwargs)
    elif allow_protocol and isinstance(x, NamedTupleInstance):
        return checksum_namedtuple(x, **kwargs)
    elif allow_protocol and isinstance(x, DataclassInstance):
        return checksum_dataclass(x, **kwargs)
    elif allow_protocol and isinstance(x, Mapping):
        return checksum_mapping(x, **kwargs)
    elif allow_protocol and isinstance(x, Iterable):
        return checksum_iterable(x, **kwargs)
    elif isinstance(x, MethodType):
        return checksum_method(x, **kwargs)
    elif isinstance(x, FunctionType):
        return checksum_function(x, **kwargs)
    elif isinstance(x, functools.partial):
        return checksum_partial(x, **kwargs)
    elif isinstance(x, type):
        return checksum_type(x, **kwargs)
    elif unk_mode == "pickle":
        return checksum_bytes(pickle.dumps(x), **kwargs)
    elif unk_mode == "error":
        msg = f"Invalid argument type {type(x)}. (expected one of {CHECKSUMABLE_TYPES})"
        raise TypeError(msg)
    else:
        raise ValueError(f"Invalid argument {unk_mode=}. (expected one of {UNK_MODES})")


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


def checksum_method(x: MethodType, **kwargs) -> int:
    fn = getattr(x.__self__, x.__name__)
    checksums = [
        checksum_any(x.__self__, **kwargs),
        checksum_function(fn, **kwargs),
    ]
    return checksum_iterable(checksums, **kwargs)


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


def checksum_partial(x: functools.partial, **kwargs) -> int:
    return checksum_iterable(("functools.partial", x.func, x.args, x.keywords))


# Intermediate functions
def checksum_builtin_number(x: BuiltinNumber, **kwargs) -> int:
    """Compute a simple checksum of a builtin scalar number."""
    # Note: instance check must follow this order: bool, int, float, complex, because isinstance(True, int) returns True !
    if isinstance(x, bool):
        return checksum_bool(x, **kwargs)
    elif isinstance(x, int):
        return checksum_int(x, **kwargs)
    elif isinstance(x, float):
        return checksum_float(x, **kwargs)
    elif isinstance(x, complex):
        return checksum_complex(x, **kwargs)
    else:
        BUILTIN_NUMBER_TYPES = ("bool", "int", "float", "complex")
        msg = (
            f"Invalid argument type {type(x)}. (expected one of {BUILTIN_NUMBER_TYPES})"
        )
        raise TypeError(msg)


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


def checksum_bytearray(x: bytearray, **kwargs) -> int:
    accumulator = kwargs.pop("accumulator", 0) + checksum_str(get_fullname(x), **kwargs)
    return _checksum_bytes_bytearray(x, accumulator=accumulator, **kwargs)


def checksum_complex(x: complex, **kwargs) -> int:
    return checksum_tensor(torch.as_tensor([x.real, x.imag]), **kwargs)


def checksum_function(x: FunctionType, **kwargs) -> int:
    return checksum_str(x.__qualname__, **kwargs)


def checksum_none(x: None, **kwargs) -> int:
    return checksum_type(x.__class__, **kwargs) + kwargs.get("accumulator", 0)


def checksum_ndarray(x: Union[np.ndarray, np.generic], **kwargs) -> int:
    if not _NUMPY_AVAILABLE:
        msg = "Cannot call function 'checksum_ndarray' because optional dependancy 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
        raise NotImplementedError(msg)

    # Supports non-numeric numpy arrays (byte string, unicode string, object, void)
    if x.dtype.kind in ("S", "U", "O", "V"):
        return checksum_any(x.tolist(), **kwargs)

    return _checksum_tensor_array_like(
        x,
        nan_to_num_fn=np.nan_to_num,
        arange_fn=np.arange,
        **kwargs,
    )


def checksum_str(x: str, **kwargs) -> int:
    return checksum_bytes(x.encode(), **kwargs)


@torch.inference_mode()
def checksum_tensor(x: Tensor, **kwargs) -> int:
    """Compute a simple checksum of a tensor. Order of values matter for the checksum."""
    return _checksum_tensor_array_like(
        x,
        nan_to_num_fn=torch.nan_to_num,
        arange_fn=torch.arange,
        **kwargs,
    )


def checksum_type(x: type, **kwargs) -> int:
    return checksum_str(x.__qualname__, **kwargs)


# Terminate functions
def checksum_bool(x: bool, **kwargs) -> int:
    return int(x) + kwargs.get("accumulator", 0)


def checksum_bytes(x: Union[bytes, bytearray], **kwargs) -> int:
    return _checksum_bytes_bytearray(x, **kwargs)


def checksum_float(x: float, **kwargs) -> int:
    xint = _interpret_float_as_int(x)
    return xint + kwargs.get("accumulator", 0)


def checksum_int(x: int, **kwargs) -> int:
    return x + kwargs.get("accumulator", 0)


def _interpret_float_as_int(x: float) -> int:
    xbytes = struct.pack("!f", x)
    xint = struct.unpack("!i", xbytes)[0]
    return xint


def _checksum_bytes_bytearray(x: Union[bytes, bytearray], **kwargs) -> int:
    return zlib.crc32(x) % (1 << 32) + kwargs.get("accumulator", 0)


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

    shape = x.shape  # type: ignore
    x = x.flatten()  # type: ignore
    arange = arange_fn(1, nelement(x) + 1, dtype=range_dtype)
    x = x + arange
    x = x * arange
    x = x.sum()

    # Ensure that accumulator exists
    kwargs["accumulator"] = kwargs.get("accumulator", 0)
    type_csum = checksum_str(get_fullname(x), **kwargs)

    kwargs["accumulator"] += type_csum
    shape_csum = checksum_iterable(shape, **kwargs)

    kwargs["accumulator"] += shape_csum
    x_csum = checksum_builtin_number(x.item(), **kwargs)

    return x_csum
