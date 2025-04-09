#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import Any, Callable, Dict
from typing import Generator as PythonGenerator
from typing import (
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    get_args,
    overload,
)

import torch
from torch import Tensor, nn
from typing_extensions import Never

from torchoutil.core.make import (
    DeviceLike,
    DTypeLike,
    GeneratorLike,
    as_device,
    as_dtype,
    as_generator,
)
from torchoutil.extras.numpy import np, numpy_view_as_complex, numpy_view_as_real
from torchoutil.nn.functional.cropping import crop_dim
from torchoutil.nn.functional.others import nelement
from torchoutil.nn.functional.padding import PadMode, PadValue, pad_dim
from torchoutil.pyoutil.collections import all_eq
from torchoutil.pyoutil.collections import flatten as builtin_flatten
from torchoutil.pyoutil.collections import prod as builtin_prod
from torchoutil.pyoutil.functools import identity  # noqa: F401
from torchoutil.pyoutil.functools import function_alias
from torchoutil.pyoutil.typing import (
    BuiltinNumber,
    BuiltinScalar,
    SizedIterable,
    T_BuiltinScalar,
    is_builtin_scalar,
)
from torchoutil.types import ComplexFloatingTensor, is_builtin_number, is_scalar_like
from torchoutil.types._typing import (
    BoolTensor0D,
    BoolTensor1D,
    BoolTensor2D,
    BoolTensor3D,
    CFloatTensor0D,
    CFloatTensor1D,
    CFloatTensor2D,
    CFloatTensor3D,
    FloatTensor0D,
    FloatTensor1D,
    FloatTensor2D,
    FloatTensor3D,
    LongTensor,
    LongTensor0D,
    LongTensor1D,
    LongTensor2D,
    LongTensor3D,
    ScalarLike,
    T_Tensor,
    T_TensorOrArray,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
)
from torchoutil.utils import return_types

T = TypeVar("T")
U = TypeVar("U")

PadCropAlign = Literal["left", "right", "center", "random"]
SqueezeMode = Literal["view_if_possible", "view", "copy", "inplace"]


def repeat_interleave_nd(x: Tensor, repeats: int, dim: int = 0) -> Tensor:
    """Generalized version of torch.repeat_interleave for N >= 1 dimensions.
    The output size will be (..., D*repeats, ...), where D is the size of the dimension of the dim argument.

    Args:
        x: Any tensor of shape (..., D, ...) with at least 1 dim.
        repeats: Number of repeats.
        dim: The dimension to repeat. defaults to 0.

    Examples::
    ----------
        >>> x = torch.as_tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        >>> repeat_interleave_nd(x, n=2, dim=0)
        tensor([[0, 1, 2, 3],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [4, 5, 6, 7]])
    """
    if x.ndim == 0:
        msg = f"Function repeat_interleave_nd does not supports 0-d tensors. (found {x.ndim=} == 0)"
        raise ValueError(msg)

    dim = dim % x.ndim
    x = x.unsqueeze(dim=dim + 1)
    shape = list(x.shape)
    shape[dim + 1] = repeats
    x = x.expand(*shape)
    x = x.flatten(dim, dim + 1)
    return x


def resample_nearest_rates(
    x: Tensor,
    rates: Union[float, Iterable[float]],
    *,
    dims: Union[int, Iterable[int]] = -1,
    round_fn: Callable[[Tensor], Tensor] = torch.floor,
) -> Tensor:
    """Nearest neigbour resampling using tensor slices.

    Args:
        x: Input tensor.
        rate: The reduction factor of each axis, e.g. a factor of 0.5 will divide the input axes by 2.
        dims: Dimensions to apply resampling. defaults to -1.
        round_fn: Rounding function to compute sub-indices. defaults to torch.floor.
    """
    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    if isinstance(rates, (int, float)):
        steps = [1.0 / rates] * len(dims)
    else:
        steps = [1.0 / rate for rate in rates]

    return resample_nearest_steps(
        x,
        steps,
        dims=dims,
        round_fn=round_fn,
    )


def resample_nearest_freqs(
    x: Tensor,
    orig_freq: int,
    new_freq: int,
    *,
    dims: Union[int, Iterable[int]] = -1,
    round_fn: Callable[[Tensor], Tensor] = torch.floor,
) -> Tensor:
    """Nearest neigbour resampling using tensor slices.

    Args:
        x: Input tensor.
        orig_freq: Source sampling rate.
        new_freq: Target sampling rate.
        dims: Dimensions to apply resampling. defaults to -1.
        round_fn: Rounding function to compute sub-indices. defaults to torch.floor.
    """
    return resample_nearest_steps(
        x,
        orig_freq / new_freq,
        dims=dims,
        round_fn=round_fn,
    )


def resample_nearest_steps(
    x: Tensor,
    steps: Union[float, Iterable[float]],
    *,
    dims: Union[int, Iterable[int]] = -1,
    round_fn: Callable[[Tensor], Tensor] = torch.floor,
) -> Tensor:
    """Nearest neigbour resampling using tensor slices.

    Args:
        x: Input tensor.
        steps: Floating step for resampling each value.
        dims: Dimensions to apply resampling. defaults to -1.
        round_fn: Rounding function to compute sub-indices. defaults to torch.floor.
    """
    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    if isinstance(steps, (int, float)):
        steps = [steps] * len(dims)
    else:
        steps = list(steps)  # type: ignore
        if len(steps) != len(dims):
            raise ValueError(f"Invalid arguments sizes {len(steps)=} != {len(dims)}.")

    slices: List[Union[slice, Tensor]] = [slice(None)] * x.ndim

    for dim, step in zip(dims, steps):
        length = x.shape[dim]
        indexes = torch.arange(0, length, step)
        indexes = round_fn(indexes).long().clamp(min=0, max=length - 1)
        slices[dim] = indexes

    x = x[slices]
    return x


def transform_drop(
    transform: Callable[[T], T],
    x: T,
    p: float,
    *,
    generator: GeneratorLike = None,
) -> T:
    """Apply a transform on a tensor with a probability of p.

    Args:
        transform: Transform to apply.
        x: Argument of the transform.
        p: Probability p to apply the transform. Cannot be negative.
            If > 1, it will apply the transform `floor(p)` times and apply a last time with a probability of `p - floor(p)`.
    """
    if p < 0.0:
        raise ValueError(f"Invalid argument {p=} < 0")

    p_floor = math.floor(p)
    for _ in range(p_floor):
        x = transform(x)

    generator = as_generator(generator)
    sampled = torch.rand((), generator=generator)
    if sampled + p_floor < p:
        x = transform(x)

    return x


def pad_and_crop_dim(
    x: Tensor,
    target_length: int,
    *,
    align: PadCropAlign = "left",
    pad_value: PadValue = 0.0,
    dim: int = -1,
    mode: PadMode = "constant",
    generator: GeneratorLike = None,
) -> Tensor:
    """Pad and crop along the specified dimension."""
    x = pad_dim(
        x,
        target_length,
        align=align,
        pad_value=pad_value,
        dim=dim,
        mode=mode,
        generator=generator,
    )
    x = crop_dim(
        x,
        target_length,
        align=align,
        dim=dim,
        generator=generator,
    )
    return x


def shuffled(
    x: Tensor,
    dims: Union[int, Iterable[int]] = -1,
    generator: GeneratorLike = None,
) -> Tensor:
    """Returns a shuffled version of the input tensor along specific dimension(s)."""
    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    generator = as_generator(generator)
    slices: List[Union[slice, Tensor]] = [slice(None) for _ in range(x.ndim)]
    for dim in dims:
        indices = torch.randperm(x.shape[dim], generator=generator)
        slices[dim] = indices
    x = x[slices]
    return x


@overload
def flatten(  # type: ignore
    x: Tensor,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> Tensor1D:
    ...


@overload
def flatten(
    x: Union[np.ndarray, np.generic],
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> np.ndarray:
    ...


@overload
def flatten(
    x: T_BuiltinScalar,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[T_BuiltinScalar]:
    ...


@overload
def flatten(  # type: ignore
    x: Iterable[T_BuiltinScalar],
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[T_BuiltinScalar]:
    ...


def flatten(
    x: Any,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> Union[Tensor1D, np.ndarray, list]:
    if isinstance(x, Tensor):
        end_dim = end_dim if end_dim is not None else x.ndim - 1
        return x.flatten(start_dim, end_dim)  # type: ignore
    elif isinstance(x, np.generic):
        return x.flatten()
    elif isinstance(x, np.ndarray):
        if start_dim == 0 and (end_dim is None or end_dim >= x.ndim - 1):
            return x.flatten()
        else:
            end_dim = end_dim if end_dim is not None else x.ndim - 1
            shape = list(x.shape)
            shape = (
                shape[:start_dim]
                + [builtin_prod(shape[start_dim : end_dim + 1])]
                + shape[end_dim + 1 :]
            )
            return x.reshape(*shape)
    else:
        return builtin_flatten(x, start_dim, end_dim, is_scalar_fn=is_scalar_like)


def squeeze(
    x: T_TensorOrArray,
    dim: Union[int, Iterable[int], None] = None,
    mode: SqueezeMode = "view_if_possible",
) -> T_TensorOrArray:
    return _squeeze_impl(x, dim, mode=mode)


def squeeze_(x: Tensor, dim: Union[int, Iterable[int], None] = None) -> Tensor:
    return _squeeze_impl(x, dim, mode="inplace")


def squeeze_copy(
    x: T_TensorOrArray,
    dim: Union[int, Iterable[int], None] = None,
) -> T_TensorOrArray:
    return _squeeze_impl(x, dim, mode="copy")


def _squeeze_impl(
    x: T_TensorOrArray,
    dim: Union[int, Iterable[int], None],
    mode: SqueezeMode,
) -> T_TensorOrArray:
    if isinstance(x, Tensor):
        return __squeeze_impl_tensor(x, dim, mode)
    elif isinstance(x, np.ndarray):
        return __squeeze_impl_array(x, dim, mode)
    else:
        msg = f"Invalid argument type {type(x)}. (expected Tensor or array)"
        raise TypeError(msg)


def __squeeze_impl_tensor(
    x: Tensor,
    dim: Union[int, Iterable[int], None],
    mode: SqueezeMode,
) -> Tensor:
    if dim is None:
        args = ()
    else:
        args = (dim,)

    if isinstance(dim, int) or dim is None:
        if mode in ("view", "view_if_possible"):
            return torch.squeeze(x, *args)
        elif mode == "copy":
            return torch.squeeze_copy(x, *args)
        elif mode == "inplace":
            return x.squeeze_(*args)
        else:
            msg = f"Invalid argument {mode=}. (expected one of {get_args(SqueezeMode)})"
            raise ValueError(msg)

    else:
        msg = f"Invalid argument type {type(dim)}. (expected int or Iterable[int] or None)"
        raise TypeError(msg)


def __squeeze_impl_array(
    x: np.ndarray,
    dim: Union[int, Iterable[int], None],
    mode: SqueezeMode,
) -> np.ndarray:
    if mode in ("view_if_possible", "copy"):
        if isinstance(dim, Iterable):
            dim = tuple(dim)
        return np.squeeze(x, axis=dim)
    else:
        msg = f"Invalid argument {mode=} with numpy array. (expected one of {('view_if_possible', 'copy')})"
        raise ValueError(msg)


def unsqueeze(
    x: T_TensorOrArray,
    dim: Union[int, Iterable[int]],
    mode: SqueezeMode = "view_if_possible",
) -> T_TensorOrArray:
    return _unsqueeze_impl(x, dim, mode=mode)


def unsqueeze_(x: Tensor, dim: Union[int, Iterable[int]]) -> Tensor:
    return _unsqueeze_impl(x, dim, mode="inplace")


def unsqueeze_copy(
    x: T_TensorOrArray,
    dim: Union[int, Iterable[int]],
) -> T_TensorOrArray:
    return _unsqueeze_impl(x, dim, mode="copy")


def _unsqueeze_impl(
    x: T_TensorOrArray,
    dim: Union[int, Iterable[int]],
    mode: SqueezeMode,
) -> T_TensorOrArray:
    if isinstance(x, Tensor):
        return __unsqueeze_impl_tensor(x, dim, mode)
    elif isinstance(x, np.ndarray):
        return __unsqueeze_impl_array(x, dim, mode)
    else:
        msg = f"Invalid argument type {type(x)}. (expected Tensor or array)"
        raise TypeError(msg)


def __unsqueeze_impl_tensor(
    x: Tensor,
    dim: Union[int, Iterable[int]],
    mode: SqueezeMode,
) -> Tensor:
    if isinstance(dim, int):
        if mode in ("view", "view_if_possible"):
            return torch.unsqueeze(x, dim)
        elif mode == "copy":
            return torch.unsqueeze_copy(x, dim)
        elif mode == "inplace":
            return x.unsqueeze_(dim)
        else:
            msg = f"Invalid argument {mode=}. (expected one of {get_args(SqueezeMode)})"
            raise ValueError(msg)

    elif isinstance(dim, Iterable):
        for dim_i in dim:
            x = __unsqueeze_impl_tensor(x, dim_i, mode)
        return x

    else:
        msg = f"Invalid argument type {type(dim)}. (expected int or Iterable[int])"
        raise TypeError(msg)


def __unsqueeze_impl_array(
    x: np.ndarray,
    dim: Union[int, Iterable[int]],
    mode: SqueezeMode,
) -> np.ndarray:
    if mode in ("view_if_possible", "copy"):
        if isinstance(dim, Iterable):
            dim = tuple(dim)
        return np.expand_dims(x, axis=dim)
    else:
        msg = f"Invalid argument {mode=} with numpy array. (expected one of {('view_if_possible', 'copy')})"
        raise ValueError(msg)


@overload
def to_item(x: T_BuiltinScalar) -> T_BuiltinScalar:
    ...


@overload
def to_item(x: Union[Tensor, np.ndarray, SizedIterable]) -> BuiltinScalar:  # type: ignore
    ...


def to_item(x: Union[ScalarLike, Tensor, np.ndarray, SizedIterable]) -> BuiltinScalar:
    """Convert scalar value to the closest built-in type."""
    if is_builtin_scalar(x, strict=True):
        return x
    elif isinstance(x, (Tensor, np.ndarray, np.generic)) and nelement(x) == 1:
        return x.item()
    elif isinstance(x, SizedIterable) and len(x) == 1:
        return to_item(next(iter(x)))
    else:
        msg = f"Invalid argument type {type(x)=}. (expected scalar-like object or an iterable containing only 1 element)"
        raise TypeError(msg)


@overload
def view_as_real(x: Tensor) -> Tensor:
    ...


@overload
def view_as_real(x: np.ndarray) -> np.ndarray:
    ...


@overload
def view_as_real(x: complex) -> Tuple[float, float]:
    ...


def view_as_real(
    x: Union[Tensor, np.ndarray, complex],
) -> Union[Tensor, np.ndarray, Tuple[float, float]]:
    """Convert complex-valued input to floating-point object."""
    if isinstance(x, Tensor):
        return torch.view_as_real(x)
    elif isinstance(x, np.ndarray):
        return numpy_view_as_real(x)
    else:
        return x.real, x.imag


@overload
def view_as_complex(x: Tensor) -> ComplexFloatingTensor:
    ...


@overload
def view_as_complex(x: np.ndarray) -> np.ndarray:
    ...


@overload
def view_as_complex(x: Tuple[float, float]) -> complex:
    ...


def view_as_complex(
    x: Union[Tensor, np.ndarray, Tuple[float, float]],
) -> Union[ComplexFloatingTensor, np.ndarray, complex]:
    """Convert floating-point input to complex-valued object."""
    if isinstance(x, Tensor):
        return torch.view_as_complex(x)  # type: ignore
    elif isinstance(x, np.ndarray):
        return numpy_view_as_complex(x)
    elif (
        isinstance(x, Sequence)
        and len(x) == 2
        and isinstance(x[0], float)
        and isinstance(x[1], float)
    ):
        return x[0] + x[1] * 1j
    else:
        raise TypeError(f"Invalid argument type {type(x)=}.")


@overload
def move_to_rec(
    x: Mapping[T, U],
    predicate: Optional[Callable[[Union[Tensor, nn.Module]], bool]] = None,
    **kwargs,
) -> Dict[T, U]:
    ...


@overload
def move_to_rec(
    x: T,
    predicate: Optional[Callable[[Union[Tensor, nn.Module]], bool]] = None,
    **kwargs,
) -> T:
    ...


def move_to_rec(
    x: Any,
    predicate: Optional[Callable[[Union[Tensor, nn.Module]], bool]] = None,
    **kwargs,
) -> Any:
    """Move all modules and tensors recursively to a specific dtype or device."""
    if "device" in kwargs:
        kwargs["device"] = as_device(kwargs["device"])

    if is_builtin_scalar(x, strict=True):
        return x
    elif isinstance(x, (Tensor, nn.Module)):
        if predicate is None or predicate(x):
            return x.to(**kwargs)
        else:
            return x
    elif isinstance(x, Mapping):
        return {k: move_to_rec(v, predicate=predicate, **kwargs) for k, v in x.items()}
    elif isinstance(x, Iterable):
        generator = (move_to_rec(xi, predicate=predicate, **kwargs) for xi in x)
        if isinstance(x, PythonGenerator):
            return generator
        elif isinstance(x, tuple):
            return tuple(generator)
        elif isinstance(x, set):
            return set(generator)
        elif isinstance(x, frozenset):
            return frozenset(generator)
        else:
            return list(generator)
    else:
        return x


# ----------
# as_tensor
# ----------


# Empty lists
@overload
def as_tensor(  # type: ignore
    data: Sequence[Never],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> Tensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Never]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> Tensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[Never]]],
    dtype: Literal[None] = None,
    device: DeviceLike = None,
) -> Tensor3D:
    ...


# bool
@overload
def as_tensor(  # type: ignore
    data: bool,
    dtype: Literal[None, "bool"] = None,
    device: DeviceLike = None,
) -> BoolTensor0D:
    ...


@overload
def as_tensor(  # type: ignore
    data: Sequence[bool],
    dtype: Literal[None, "bool"] = None,
    device: DeviceLike = None,
) -> BoolTensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[bool]],
    dtype: Literal[None, "bool"] = None,
    device: DeviceLike = None,
) -> BoolTensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[bool]]],
    dtype: Literal[None, "bool"] = None,
    device: DeviceLike = None,
) -> BoolTensor3D:
    ...


# int
@overload
def as_tensor(
    data: int,
    dtype: Literal[None, "int64", "long"] = None,
    device: DeviceLike = None,
) -> LongTensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[int],
    dtype: Literal[None, "int64", "long"] = None,
    device: DeviceLike = None,
) -> LongTensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[int]],
    dtype: Literal[None, "int64", "long"] = None,
    device: DeviceLike = None,
) -> LongTensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[int]]],
    dtype: Literal[None, "int64", "long"] = None,
    device: DeviceLike = None,
) -> LongTensor3D:
    ...


# float
@overload
def as_tensor(
    data: float,
    dtype: Literal[None, "float32", "float"] = None,
    device: DeviceLike = None,
) -> FloatTensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[float],
    dtype: Literal[None, "float32", "float"] = None,
    device: DeviceLike = None,
) -> FloatTensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[float]],
    dtype: Literal[None, "float32", "float"] = None,
    device: DeviceLike = None,
) -> FloatTensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[float]]],
    dtype: Literal[None, "float32", "float"] = None,
    device: DeviceLike = None,
) -> FloatTensor3D:
    ...


# complex
@overload
def as_tensor(
    data: complex,
    dtype: Literal[None, "complex64", "cfloat"] = None,
    device: DeviceLike = None,
) -> CFloatTensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[complex],
    dtype: Literal[None, "complex64", "cfloat"] = None,
    device: DeviceLike = None,
) -> CFloatTensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[complex]],
    dtype: Literal[None, "complex64", "cfloat"] = None,
    device: DeviceLike = None,
) -> CFloatTensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[complex]]],
    dtype: Literal[None, "complex64", "cfloat"] = None,
    device: DeviceLike = None,
) -> CFloatTensor3D:
    ...


# BuiltinNumber
@overload
def as_tensor(
    data: BuiltinNumber,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor0D:
    ...


@overload
def as_tensor(
    data: Sequence[BuiltinNumber],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor1D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[BuiltinNumber]],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor2D:
    ...


@overload
def as_tensor(
    data: Sequence[Sequence[Sequence[BuiltinNumber]]],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor3D:
    ...


@overload
def as_tensor(
    data: Any,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    ...


def as_tensor(
    data: Any,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Convert arbitrary data to tensor.

    Unlike `torch.as_tensor`, it works recursively and stack sequences like List[Tensor]. It also accept python generator objects.

    Args:
        data: Data to convert to tensor. Can be Tensor, np.ndarray, list, tuple or any number-like object.
        dtype: Target torch dtype. defaults to None.
        device: Target torch device. defaults to None.

    Returns:
        PyTorch tensor created from data.
    """
    if isinstance(data, (Tensor, np.ndarray, np.number)) or is_builtin_number(data):
        dtype = as_dtype(dtype)
        device = as_device(device)
        return torch.as_tensor(data, dtype=dtype, device=device)

    elif isinstance(data, (list, tuple, PythonGenerator)):
        dtype = as_dtype(dtype)
        device = as_device(device)

        tensors: List[Tensor] = [
            as_tensor(data_i, dtype=dtype, device=device) for data_i in data
        ]
        if len(tensors) == 0:
            return torch.as_tensor(tensors, dtype=dtype, device=device)

        shapes = [tensor.shape for tensor in tensors]
        if not all_eq(shapes):
            uniq_shapes = tuple(set(shapes))
            msg = f"Cannot convert to tensor a list of elements with heterogeneous shapes. (found different shapes: {uniq_shapes})"
            raise ValueError(msg)

        return torch.stack(tensors)

    else:
        EXPECTED = (
            Tensor,
            np.ndarray,
            np.number,
            BuiltinNumber,
            list,
            tuple,
            PythonGenerator,
        )
        msg = f"Invalid argument type '{type(data)}'. (expected one of {EXPECTED})"
        raise TypeError(msg)


@overload
def topk(
    x: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: Literal[True] = True,
    return_indices: Literal[True] = True,
) -> return_types.topk:
    ...


@overload
def topk(
    x: T_Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: Literal[True] = True,
    return_indices: Literal[False],
) -> T_Tensor:
    ...


@overload
def topk(
    x: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: Literal[False],
    return_indices: Literal[True] = True,
) -> LongTensor:
    ...


@overload
def topk(
    x: T_Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: bool = True,
    return_indices: bool = True,
) -> Union[T_Tensor, LongTensor, return_types.topk]:
    ...


def topk(
    x: T_Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    return_values: bool = True,
    return_indices: bool = True,
) -> Union[T_Tensor, LongTensor, return_types.topk]:
    result = x.topk(
        k=k,
        dim=dim,
        largest=largest,
        sorted=sorted,
    )
    if return_values and return_indices:
        return result  # type: ignore
    elif return_values:
        return result.values  # type: ignore
    elif return_indices:
        return result.indices  # type: ignore
    else:
        msg = f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
        raise ValueError(msg)


@overload
def top_p(
    x: Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: Literal[True] = True,
    return_indices: Literal[True] = True,
) -> return_types.top_p:
    ...


@overload
def top_p(
    x: T_Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: Literal[True] = True,
    return_indices: Literal[False],
) -> T_Tensor:
    ...


@overload
def top_p(
    x: Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: Literal[False],
    return_indices: Literal[True] = True,
) -> LongTensor:
    ...


@overload
def top_p(
    x: T_Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: bool = True,
    return_indices: bool = True,
) -> Union[T_Tensor, LongTensor, return_types.top_p]:
    ...


def top_p(
    x: T_Tensor,
    p: float,
    dim: int = -1,
    largest: bool = True,
    *,
    return_values: bool = True,
    return_indices: bool = True,
) -> Union[T_Tensor, LongTensor, return_types.top_p]:
    values, indices = x.sort(dim=dim, descending=largest)
    cumulated = torch.cumsum(values, dim=dim)
    idx = (cumulated >= p).long().argmax()
    result = return_types.top_p([values[:idx], indices[:idx]])

    if return_values and return_indices:
        return result  # type: ignore
    elif return_values:
        return result.values  # type: ignore
    elif return_indices:
        return result.indices  # type: ignore
    else:
        msg = f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
        raise ValueError(msg)


@function_alias(as_tensor)
def to_tensor(*args, **kwargs):
    ...


@function_alias(topk)
def top_k(*args, **kwargs):
    ...
