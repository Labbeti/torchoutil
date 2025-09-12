#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import itertools
import math
import pickle
import re
import struct
import zlib
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from types import FunctionType, MethodType
from typing import Callable, Iterable, Literal, Mapping, Union, get_args

import torch
from torch import Tensor, nn
from typing_extensions import TypeAlias

from torchoutil.core.packaging import _NUMPY_AVAILABLE, _PANDAS_AVAILABLE
from torchoutil.extras.numpy import np
from torchoutil.nn.functional.predicate import is_complex, is_floating_point
from torchoutil.pyoutil.functools import function_alias
from torchoutil.pyoutil.importlib import Placeholder
from torchoutil.pyoutil.inspect import get_fullname
from torchoutil.pyoutil.typing import (
    BuiltinNumber,
    BuiltinScalar,
    DataclassInstance,
    NamedTupleInstance,
    NoneType,
)

if _PANDAS_AVAILABLE:
    import pandas as pd

    DataFrame = pd.DataFrame  # type: ignore
else:

    class DataFrame(Placeholder):
        ...


Checksumable: TypeAlias = Union[
    int,
    bool,
    complex,
    float,
    NoneType,
    str,
    bytes,
    bytearray,
    re.Pattern,
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
    slice,
    Path,
]
UnkMode = Literal["pickle", "error"]


# Recursive functions for union of types
def checksum_any(
    x: Checksumable,
    *,
    unk_mode: UnkMode = "error",
    allow_protocol: bool = True,
    **kwargs,
) -> int:
    """Compute checksum of an arbitrary python object.

    The property of a checksum is: for all any supported objects a and b, `(a == b) => (checksum(a) == checksum(b))`.
    This function is deterministic across executions by default.

    Args:
        x: Object to checksum.
        unk_mode: Defines behaviour when x is not a supported type OR contains elements that are not supported.
            "error": raises a TypeError.
            "pickle": convert object to bytes using pickle module. However, this conversion depends of the object implementation and might be not deterministic.
            defaults to "error".
        allow_protocol: Whether or not accept to use duck typing to detect NamedTuples, Dataclasses, Mappings or Iterables. defaults to True.
        **kwargs: Optional arguments to customize object checksum.
    """
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
    elif isinstance(x, slice):
        return checksum_slice(x, **kwargs)
    elif isinstance(x, Path):
        return checksum_path(x, **kwargs)
    elif isinstance(x, re.Pattern):
        return checksum_pattern(x, **kwargs)
    elif isinstance(x, nn.Module):
        return checksum_module(x, **kwargs)
    elif isinstance(x, Tensor):
        return checksum_tensor(x, **kwargs)
    elif _NUMPY_AVAILABLE and isinstance(x, (np.ndarray, np.generic)):
        return checksum_ndarray(x, **kwargs)
    elif isinstance(x, torch.dtype) or (_NUMPY_AVAILABLE and isinstance(x, np.dtype)):
        return checksum_dtype(x, **kwargs)
    elif _PANDAS_AVAILABLE and isinstance(x, DataFrame):
        return checksum_dataframe(x, **kwargs)
    elif allow_protocol and isinstance(x, NamedTupleInstance):
        return checksum_namedtuple(x, **kwargs)
    elif allow_protocol and isinstance(x, DataclassInstance):
        return checksum_dataclass(x, **kwargs)
    elif (allow_protocol and isinstance(x, Mapping)) or isinstance(x, dict):
        return checksum_mapping(x, **kwargs)
    elif (allow_protocol and isinstance(x, Iterable)) or isinstance(
        x, (list, tuple, set, frozenset, range)
    ):
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
        msg = f"Invalid argument type {type(x)}. (expected one of {get_args(Checksumable)})"
        raise TypeError(msg)
    else:
        msg = f"Invalid argument {unk_mode=}. (expected one of {get_args(UnkMode)})"
        raise ValueError(msg)


@function_alias(checksum_any)
def checksum(*args, **kwargs):
    ...


def checksum_dataclass(x: DataclassInstance, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_mapping(asdict(x), **kwargs)


def checksum_dataframe(x: DataFrame, **kwargs) -> int:
    if not _PANDAS_AVAILABLE:
        msg = "Cannot call function 'checksum_dataframe' because optional dependency 'pandas' is not installed. Please install it using 'pip install torchoutil[extras]'"
        raise NotImplementedError(msg)

    # hash_value = hashlib.sha1(pd.util.hash_pandas_object(x).values).hexdigest()
    # csum = checksum_str(hash_value, **kwargs)
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    x = x.to_dict()
    return checksum_mapping(x, **kwargs)  # type: ignore


def checksum_dtype(x: Union[torch.dtype, np.dtype], **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    xstr = str(x)
    return checksum_str(xstr, **kwargs)


def checksum_iterable(x: Iterable, **kwargs) -> int:
    accumulator = kwargs.pop("accumulator", 0) + __cached_checksum_str(get_fullname(x))
    csum = sum(
        checksum_any(xi, accumulator=accumulator + (i + 1), **kwargs) * (i + 1)
        for i, xi in enumerate(x)
    )
    return csum + accumulator


def checksum_mapping(x: Mapping, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_iterable(x.items(), **kwargs)


def checksum_method(x: MethodType, **kwargs) -> int:
    fn = getattr(x.__self__, x.__name__)
    checksums = [
        checksum_any(x.__self__, **kwargs),  # type: ignore
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
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_mapping(x._asdict(), **kwargs)


def checksum_partial(x: functools.partial, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_iterable((x.func, x.args, x.keywords), **kwargs)


def checksum_pattern(x: re.Pattern, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_str(str(x), **kwargs)


def checksum_path(x: Path, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_str(str(x), **kwargs)


def checksum_slice(x: slice, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_iterable((x.start, x.stop, x.step), **kwargs)


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
        msg = f"Invalid argument type {type(x)}. (expected one of {get_args(BuiltinNumber)})"
        raise TypeError(msg)


def checksum_builtin_scalar(x: BuiltinScalar, **kwargs) -> int:
    if isinstance(x, BuiltinNumber):
        return checksum_builtin_number(x, **kwargs)
    elif isinstance(x, bytes):
        return checksum_bytes(x, **kwargs)
    elif x is None:
        return checksum_none(x, **kwargs)
    elif isinstance(x, str):
        return checksum_str(x, **kwargs)
    else:
        msg = f"Invalid argument type {type(x)}. (expected one of {get_args(BuiltinScalar)})"
        raise TypeError(msg)


def checksum_bytearray(x: bytearray, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return _checksum_bytes_bytearray(x, **kwargs)


def checksum_bytes(x: Union[bytes, bytearray], **kwargs) -> int:
    return _checksum_bytes_bytearray(x, **kwargs)


def checksum_complex(x: complex, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_tensor(torch.as_tensor([x.real, x.imag]), **kwargs)


def checksum_function(x: FunctionType, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_str(x.__qualname__, **kwargs)


def checksum_none(x: None, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_type(x.__class__, **kwargs) + kwargs.get("accumulator", 0)


def checksum_ndarray(x: Union[np.ndarray, np.generic], **kwargs) -> int:
    if not _NUMPY_AVAILABLE:
        msg = "Cannot call function 'checksum_ndarray' because optional dependency 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
        raise NotImplementedError(msg)

    # Supports non-numeric numpy arrays (byte string, unicode string, object, void)
    if x.dtype.kind in ("S", "U", "O", "V"):
        return checksum_any(x.tolist(), **kwargs)

    return _checksum_tensor_array_like(
        x,
        nan_to_num_fn=np.nan_to_num,
        **kwargs,
    )


def checksum_str(x: str, **kwargs) -> int:
    kwargs["accumulator"] = kwargs.get("accumulator", 0) + __cached_checksum_str(
        get_fullname(x)
    )
    return checksum_bytes(x.encode(), **kwargs)


@torch.inference_mode()
def checksum_tensor(x: Tensor, **kwargs) -> int:
    """Compute a simple checksum of a tensor. Order of values matter for the checksum."""
    return _checksum_tensor_array_like(
        x,
        nan_to_num_fn=torch.nan_to_num,
        **kwargs,
    )


def checksum_type(x: type, **kwargs) -> int:
    return checksum_str(x.__qualname__, **kwargs)


# Terminate functions
def checksum_bool(x: bool, **kwargs) -> int:
    xint = int(x)
    return __terminate_checksum(
        xint,
        get_fullname(x),
        **kwargs,
    )


def checksum_float(x: float, **kwargs) -> int:
    xint = _interpret_float_as_int(x)
    return __terminate_checksum(
        xint,
        get_fullname(x),
        **kwargs,
    )


def checksum_int(x: int, **kwargs) -> int:
    xint = x
    return __terminate_checksum(
        xint,
        get_fullname(x),
        **kwargs,
    )


def _checksum_bytes_bytearray(x: Union[bytes, bytearray], **kwargs) -> int:
    xint = zlib.crc32(x) % (1 << 32)
    return __terminate_checksum(
        xint,
        get_fullname(x),
        **kwargs,
    )


def __terminate_checksum(x: int, fullname: str, **kwargs) -> int:
    return x + __cached_checksum_str(fullname) + kwargs.get("accumulator", 0)


@lru_cache(maxsize=None)
def __cached_checksum_str(x: str) -> int:
    return zlib.crc32(x.encode()) % (1 << 32)


def _interpret_float_as_int(x: float) -> int:
    xbytes = struct.pack(">d", x)
    xint = struct.unpack(">q", xbytes)[0]
    return xint


def _checksum_tensor_array_like(
    x: Union[Tensor, np.ndarray, np.generic],
    *,
    nan_to_num_fn: Callable,
    **kwargs,
) -> int:
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

    # Ensure that accumulator exists
    kwargs["accumulator"] = kwargs.get("accumulator", 0)

    kwargs["accumulator"] += checksum_dtype(x.dtype, **kwargs)
    kwargs["accumulator"] += checksum_iterable(x.shape, **kwargs)
    kwargs["accumulator"] += __cached_checksum_str(get_fullname(x))

    if isinstance(x, (np.ndarray, np.generic)):
        xbytes = x.tobytes()
        csum = checksum_bytes(xbytes, **kwargs)
    elif isinstance(x, Tensor):
        if _NUMPY_AVAILABLE:
            xbytes = x.cpu().numpy().tobytes()
        else:
            xbytes = _serialize_tensor_to_bytes(x)
        csum = checksum_bytes(xbytes, **kwargs)
    else:
        msg = f"invalid argument type {type(x)}. (expected ndarray or Tensor)"
        raise TypeError(msg)

    return csum


def _serialize_tensor_to_bytes(x: Tensor) -> bytes:
    """Convert tensor data to bytes, but very slow compare to numpy' tobytes() method."""
    x = x.view(torch.int8).view(-1)
    xbytes = struct.pack(f"{len(x)}b", *x)
    return xbytes
