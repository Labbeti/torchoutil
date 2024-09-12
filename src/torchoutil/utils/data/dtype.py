#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Generic, Iterable, Tuple, TypeVar, Union

import torch
from torch import Tensor

import pyoutil as po
import torchoutil as to
from torchoutil.types import ACCEPTED_NUMPY_DTYPES, BuiltinScalar, np

T_Invalid = TypeVar("T_Invalid")
T_Empty = TypeVar("T_Empty")
T_EmptyNp = TypeVar("T_EmptyNp")
T_EmptyTorch = TypeVar("T_EmptyTorch")


# return type for torch_dtype when an invalid data is passed as argument, like str
class InvalidTorchDType(metaclass=po.Singleton):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass(frozen=True)
class ShapeDTypeInfo(Generic[T_Invalid, T_EmptyNp, T_EmptyTorch]):
    shape: Tuple[int, ...]
    torch_dtype: Union[torch.dtype, T_Invalid, T_EmptyTorch]
    numpy_dtype: Union[np.dtype, T_EmptyNp]

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


def get_default_numpy_dtype() -> np.dtype:
    return np.empty((0,)).dtype


def scan_shape_dtypes(x: Any) -> ShapeDTypeInfo[InvalidTorchDType, np.dtype, None]:
    """Returns the shape and the hdf_dtype for an input."""
    shape = to.shape(x)
    torch_dtype = scan_torch_dtype(x)
    numpy_dtype = scan_numpy_dtype(x)
    info = ShapeDTypeInfo[InvalidTorchDType, np.dtype, None](
        shape,
        torch_dtype,
        numpy_dtype,
    )
    return info


def scan_torch_dtype(
    x: Any,
    *,
    invalid: T_Invalid = InvalidTorchDType(),
    empty: T_EmptyNp = None,
) -> Union[torch.dtype, T_Invalid, T_EmptyNp]:
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

    if isinstance(x, (str, bytes)):
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

    if isinstance(x, (str, bytes)):
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


def torch_dtype_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
    x = torch.empty((0,), dtype=dtype)
    x = to.tensor_to_numpy(x)
    return x.dtype


def numpy_dtype_to_torch_dtype(
    dtype: np.dtype,
    *,
    invalid: T_Invalid = InvalidTorchDType(),
) -> Union[torch.dtype, T_Invalid]:
    x = np.empty((0,), dtype=dtype)
    if x.dtype not in ACCEPTED_NUMPY_DTYPES:
        return invalid
    else:
        x = to.numpy_to_tensor(x)
        return x.dtype


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
        raise ValueError(f"Invalid argument {dtype=}.")
