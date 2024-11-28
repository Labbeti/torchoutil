#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, Iterable, Tuple, TypeVar, Union

import torch
from torch import Tensor

import torchoutil as to
from torchoutil import pyoutil as po
from torchoutil.extras.numpy.definitions import ACCEPTED_NUMPY_DTYPES, np
from torchoutil.pyoutil import BuiltinScalar, get_current_fn_name

T_Invalid = TypeVar("T_Invalid", covariant=True)
T_EmptyNp = TypeVar("T_EmptyNp", covariant=True)
T_EmptyTorch = TypeVar("T_EmptyTorch", covariant=True)


class InvalidTorchDType(metaclass=po.Singleton):
    """Default return type for torch_dtype when an invalid data is passed as argument of scan_torch_dtype function. (like str for example)"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass(frozen=True)
class ShapeDTypeInfo(Generic[T_Invalid, T_EmptyTorch, T_EmptyNp]):
    shape: Tuple[int, ...]
    torch_dtype: Union[torch.dtype, T_Invalid, T_EmptyTorch]
    numpy_dtype: Union[np.dtype, T_EmptyNp]
    valid_shape: bool

    @property
    def fill_value(self) -> BuiltinScalar:
        return numpy_dtype_to_fill_value(self.numpy_dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def kind(self) -> str:
        if isinstance(self.numpy_dtype, np.dtype):
            return self.numpy_dtype.kind
        else:
            return "V"


def scan_shape_dtypes(
    x: Any,
    *,
    accept_heterogeneous_shape: bool = False,
    empty_torch: T_EmptyTorch = None,
    empty_np: T_EmptyNp = np.dtype("V"),
) -> ShapeDTypeInfo[InvalidTorchDType, T_EmptyTorch, T_EmptyNp]:
    """Returns the shape and the hdf_dtype for an input."""
    valid_shape, shape = to.shape(x, return_valid=True)
    if not accept_heterogeneous_shape and not valid_shape:
        msg = f"Invalid argument {x} for {get_current_fn_name()}. (cannot compute shape for heterogeneous data)"
        raise ValueError(msg)

    torch_dtype = scan_torch_dtype(x, empty=empty_torch)
    numpy_dtype = scan_numpy_dtype(x, empty=empty_np)

    info = ShapeDTypeInfo[InvalidTorchDType, T_EmptyTorch, T_EmptyNp](
        shape,
        torch_dtype,
        numpy_dtype,
        valid_shape,
    )
    return info


def scan_torch_dtype(
    x: Any,
    *,
    invalid: T_Invalid = InvalidTorchDType(),
    empty: T_EmptyTorch = None,
) -> Union[torch.dtype, T_Invalid, T_EmptyTorch]:
    """Returns torch dtype of an arbitrary object. Works recursively on tuples and lists. An instance of InvalidTorchDType can be returned if a str is passed."""
    if isinstance(x, (int, float, bool, complex)):
        torch_dtype = torch.as_tensor(x).dtype
        return torch_dtype

    if isinstance(x, Tensor):
        torch_dtype = x.dtype
        return torch_dtype

    if isinstance(x, (np.ndarray, np.generic)):
        torch_dtype = numpy_dtype_to_torch_dtype(x.dtype, invalid=invalid)
        return torch_dtype

    if isinstance(x, (str, bytes, bytearray)):
        return invalid

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return empty

        torch_dtypes = [scan_torch_dtype(xi, invalid=invalid, empty=empty) for xi in x]
        torch_dtype = merge_torch_dtypes(torch_dtypes, invalid=invalid, empty=empty)
        return torch_dtype

    msg = f"Unsupported type {x.__class__.__name__} in function {po.get_current_fn_name()}."
    raise TypeError(msg)


def scan_numpy_dtype(
    x: Any,
    *,
    empty: T_EmptyNp = np.dtype("V"),
) -> Union[np.dtype, T_EmptyNp]:
    if isinstance(x, (int, float, bool, complex)):
        numpy_dtype = np.array(x).dtype
        return numpy_dtype

    if isinstance(x, Tensor):
        numpy_dtype = torch_dtype_to_numpy_dtype(x.dtype)
        return numpy_dtype

    if isinstance(x, (np.ndarray, np.generic)):
        numpy_dtype = x.dtype
        return numpy_dtype

    if isinstance(x, (str, bytes, bytearray)):
        numpy_dtype = np.array(x).dtype
        return numpy_dtype

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return empty

        numpy_dtypes = [scan_numpy_dtype(xi, empty=empty) for xi in x]
        numpy_dtype = merge_numpy_dtypes(numpy_dtypes, empty=empty)
        return numpy_dtype

    msg = f"Unsupported type {x.__class__.__name__} in function {po.get_current_fn_name()}."
    raise TypeError(msg)


@lru_cache(maxsize=None)
def torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    x = torch.empty((0,), dtype=dtype)
    x = to.tensor_to_numpy(x)
    return x.dtype


@lru_cache(maxsize=None)
def numpy_dtype_to_torch_dtype(
    dtype: np.dtype,
    *,
    invalid: T_Invalid = InvalidTorchDType(),
) -> Union[torch.dtype, T_Invalid]:
    if dtype in ACCEPTED_NUMPY_DTYPES:
        x = np.empty((0,), dtype=dtype)
        x = to.numpy_to_tensor(x)
        return x.dtype
    else:
        return invalid


def numpy_dtype_to_fill_value(dtype: Any) -> BuiltinScalar:
    if not isinstance(dtype, np.dtype):
        return None

    kind = dtype.kind
    if kind in ("b",):
        return False
    elif kind in ("u", "i"):
        return 0
    elif kind in ("f",):
        return 0.0
    elif kind in ("c",):
        return 0j
    elif kind in ("U", "S"):
        return ""
    else:
        KINDS = ("b", "u", "i", "f", "c", "U", "S")
        msg = f"Invalid argument {dtype=}. (expected dtype.kind in {KINDS})"
        raise ValueError(msg)


def merge_numpy_dtypes(
    dtypes: Iterable[Union[np.dtype, T_EmptyNp]],
    *,
    empty: T_EmptyNp = np.dtype("V"),
) -> Union[np.dtype, T_EmptyNp]:
    dtypes = list(dict.fromkeys(dtypes))
    dtypes = [dtype for dtype in dtypes if dtype != empty]
    if len(dtypes) == 0:
        return empty

    dummy_arrays = [np.empty((0,), dtype=dtype) for dtype in dtypes]  # type: ignore
    dtype = np.stack(dummy_arrays).dtype
    return dtype


def merge_torch_dtypes(
    dtypes: Iterable[Union[torch.dtype, T_Invalid, T_EmptyNp]],
    *,
    invalid: T_Invalid = InvalidTorchDType(),
    empty: T_EmptyNp = None,
) -> Union[torch.dtype, T_Invalid, T_EmptyNp]:
    dtypes = list(dict.fromkeys(dtypes))
    dtypes = [dtype for dtype in dtypes if dtype != empty]
    if len(dtypes) == 0:
        return empty
    if any(dtype == invalid for dtype in dtypes):
        return invalid

    dummy_tensors = [torch.empty((0,), dtype=dtype) for dtype in dtypes]  # type: ignore
    dtype = torch.stack(dummy_tensors).dtype
    return dtype


def get_default_numpy_dtype() -> np.dtype:
    return np.empty((0,)).dtype
