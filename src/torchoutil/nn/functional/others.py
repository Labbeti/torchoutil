#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from typing import Any, Callable, Dict
from typing import Generator as PythonGenerator
from typing import (
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Tensor, nn
from typing_extensions import TypeGuard

from torchoutil.core.get import get_device
from torchoutil.extras.numpy import (
    ACCEPTED_NUMPY_DTYPES,
    np,
    numpy_all_eq,
    numpy_all_ne,
    numpy_is_complex,
    numpy_is_floating_point,
    numpy_view_as_complex,
    numpy_view_as_real,
)
from torchoutil.pyoutil.collections import all_eq as builtin_all_eq
from torchoutil.pyoutil.collections import all_ne as builtin_all_ne
from torchoutil.pyoutil.collections import is_sorted as builtin_is_sorted
from torchoutil.pyoutil.collections import prod as builtin_prod
from torchoutil.pyoutil.collections import unzip
from torchoutil.pyoutil.functools import identity
from torchoutil.pyoutil.typing import (
    BuiltinScalar,
    SizedIterable,
    T_BuiltinNumber,
    is_builtin_number,
    is_builtin_scalar,
)
from torchoutil.types._typing import (
    ComplexFloatingTensor,
    FloatingTensor,
    LongTensor,
    ScalarLike,
    T_TensorLike,
)
from torchoutil.types.guards import is_list_tensor, is_scalar_like, is_tuple_tensor
from torchoutil.utils import return_types

T = TypeVar("T")
U = TypeVar("U")


def count_parameters(
    model: nn.Module,
    *,
    recurse: bool = True,
    only_trainable: bool = False,
    buffers: bool = False,
) -> int:
    """Returns the number of parameters in a module."""
    params = (
        param
        for param in model.parameters(recurse)
        if not only_trainable or param.requires_grad
    )

    if buffers:
        params = itertools.chain(params, (buffer for buffer in model.buffers(recurse)))

    num_params = sum(param.numel() for param in params)
    return num_params


def find(
    x: Tensor,
    value: Any,
    default: Union[None, Tensor, int, float] = None,
    dim: int = -1,
) -> LongTensor:
    """Return the index of the first occurrence of value in a tensor."""
    if x.ndim == 0:
        msg = f"Function find does not supports 0-d tensors. (found {x.ndim=} == 0)"
        raise ValueError(msg)

    mask = x.eq(value)
    contains = mask.any(dim=dim)
    indices = mask.long().argmax(dim=dim)

    if default is None:
        if not contains.all():
            raise RuntimeError(f"Cannot find {value=} in tensor.")
        return indices  # type: ignore
    else:
        output = torch.where(contains, indices, default)
        return output  # type: ignore


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
        kwargs["device"] = get_device(kwargs["device"])

    if isinstance(x, (str, float, int, bool, complex)):
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
        else:
            return list(generator)
    else:
        return x


def can_be_stacked(
    tensors: Union[List[Any], Tuple[Any, ...]],
) -> TypeGuard[Union[List[Tensor], Tuple[Tensor, ...]]]:
    """Returns True if inputs can be passed to `torch.stack` function, i.e. contanis a list or tuple of tensors with the same shape.

    Alias of :func:`~torchoutil.nn.functional.others.is_stackable`.
    """
    return is_stackable(tensors)


def is_stackable(
    tensors: Union[List[Any], Tuple[Any, ...]],
) -> TypeGuard[Union[List[Tensor], Tuple[Tensor, ...]]]:
    """Returns True if inputs can be passed to `torch.stack` function, i.e. contanis a list or tuple of tensors with the same shape."""
    if not is_list_tensor(tensors) and not is_tuple_tensor(tensors):
        return False
    if len(tensors) == 0:
        return False
    shape0 = tensors[0].shape
    result = all(tensor.shape == shape0 for tensor in tensors[1:])
    return result


def can_be_converted_to_tensor(x: Any) -> bool:
    """Returns True if inputs can be passed to `torch.as_tensor` function.

    Alias of :func:`~torchoutil.nn.functional.others.is_convertible_to_tensor`.

    This function returns False for heterogeneous inputs like `[[], 1]`, but this kind of value can be accepted by `torch.as_tensor`.
    """
    return is_convertible_to_tensor(x)


def is_convertible_to_tensor(x: Any) -> bool:
    """Returns True if inputs can be passed to `torch.as_tensor` function.

    This function returns False for heterogeneous inputs like `[[], 1]`, but this kind of value can be accepted by `torch.as_tensor`.
    """
    if isinstance(x, Tensor):
        return True
    else:
        return __can_be_converted_to_tensor_nested(x)


@overload
def ndim(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    return_valid: Literal[False] = False,
) -> int:
    ...


@overload
def ndim(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    return_valid: Literal[True],
) -> return_types.ndim:
    ...


def ndim(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    return_valid: bool = False,
    use_first_for_list_tuple: bool = False,
) -> Union[int, return_types.ndim]:
    """Scan first argument to return its number of dimension(s). Works recursively with Tensors, numpy arrays and builtins types instances.

    Note: Sets and dicts are considered as scalars with a shape equal to 0.

    Args:
        x: Input value to scan.
        return_valid: If True, returns a tuple containing a boolean indicator if the data has an homogeneous ndim instead of raising a ValueError.
        use_first_for_list_tuple: If True, use first value to determine ndim for list and tuple argument. Otherwise it will scan each value in argument to determine its shape. defaults to False.

    Raises:
        ValueError if input has an heterogeneous number of dimensions.
        TypeError if input has an unsupported type.
    """

    def _impl(
        x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    ) -> Tuple[bool, int]:
        if is_scalar_like(x):
            return True, 0
        elif isinstance(x, (Tensor, np.ndarray, np.generic)):
            return True, x.ndim
        elif isinstance(x, (set, frozenset, dict)):
            return True, 0
        elif isinstance(x, (list, tuple)):
            valids_and_ndims = unzip(_impl(xi) for xi in x)  # type: ignore
            if len(valids_and_ndims) == 0:
                return True, 1

            valids, ndims = valids_and_ndims
            if (use_first_for_list_tuple and valids[0]) or (
                all(valids) and builtin_all_eq(ndims)
            ):
                return True, ndims[0] + 1
            else:
                return False, -1
        else:
            raise TypeError(f"Invalid argument type {type(x)}.")

    valid, ndim = _impl(x)
    if return_valid:
        return return_types.ndim(valid, ndim)
    elif valid:
        return ndim
    else:
        msg = f"Invalid argument {x}. (cannot compute ndim for heterogeneous data)"
        raise ValueError(msg)


@overload
def shape(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    output_type: Callable[[Tuple[int, ...]], T] = identity,
    return_valid: Literal[False] = False,
) -> T:
    ...


@overload
def shape(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    output_type: Callable[[Tuple[int, ...]], T] = identity,
    return_valid: Literal[True],
) -> return_types.shape[T]:
    ...


def shape(
    x: Union[ScalarLike, Tensor, np.ndarray, Iterable],
    *,
    output_type: Callable[[Tuple[int, ...]], T] = identity,
    return_valid: bool = False,
    use_first_for_list_tuple: bool = False,
) -> Union[T, return_types.shape[T]]:
    """Scan first argument to return its shape. Works recursively with Tensors, numpy arrays and builtins types instances.

    Note: Sets and dicts are considered as scalars with a shape equal to ().

    Args:
        x: Input value to scan.
        output_type: Output shape type. defaults to identity, which returns a tuple of ints.
        return_valid: If True, returns a tuple containing a boolean indicator if the data has an homogeneous shape instead of raising a ValueError.
        use_first_for_list_tuple: If True, use first value to determine ndim for list and tuple argument. Otherwise it will scan each value in argument to determine its shape. defaults to False.

    Raises:
        ValueError: if input has an heterogeneous shape.
        TypeError: if input has an unsupported type.
    """

    def _impl(
        x: Union[ScalarLike, Tensor, np.ndarray, Iterable]
    ) -> Tuple[bool, Tuple[int, ...]]:
        if is_scalar_like(x):
            return True, ()
        elif isinstance(x, (Tensor, np.ndarray, np.generic)):
            return True, tuple(x.shape)
        elif isinstance(x, (set, frozenset, dict)):
            return True, ()
        elif isinstance(x, (list, tuple)):
            valids_and_shapes = unzip(_impl(xi) for xi in x)  # type: ignore
            if len(valids_and_shapes) == 0:
                return True, (0,)

            valids, shapes = valids_and_shapes
            if (use_first_for_list_tuple and valids[0]) or (
                all(valids) and builtin_all_eq(shapes)
            ):
                return True, (len(shapes),) + shapes[0]
            else:
                return False, ()
        else:
            raise TypeError(f"Invalid argument type {type(x)}.")

    valid, shape = _impl(x)
    if return_valid:
        shape = output_type(shape)
        return return_types.shape(valid, shape)
    elif valid:
        shape = output_type(shape)
        return shape
    else:
        msg = f"Invalid argument {x}. (cannot compute shape for heterogeneous data)"
        raise ValueError(msg)


def to_item(x: Union[ScalarLike, Tensor, np.ndarray, SizedIterable]) -> BuiltinScalar:
    """Convert scalar value to built-in type."""
    if is_builtin_scalar(x, strict=True):
        return x
    elif isinstance(x, (Tensor, np.ndarray, np.generic)) and nelement(x) == 1:
        return x.item()
    elif isinstance(x, SizedIterable) and len(x) == 1:
        return to_item(next(iter(x)))
    else:
        msg = f"Invalid argument type {type(x)=}. (expected scalar-like object)"
        raise TypeError(msg)


def ranks(x: Tensor, dim: int = -1, descending: bool = False) -> Tensor:
    """Get the ranks of each value in range [0, x.shape[dim][."""
    return x.argsort(dim, descending).argsort(dim)


def nelement(x: Union[ScalarLike, Tensor, np.ndarray, Iterable]) -> int:
    """Returns the number of elements in Tensor-like object."""
    if isinstance(x, Tensor):
        return x.nelement()
    elif isinstance(x, (np.ndarray, np.generic)):
        return x.size
    else:
        return builtin_prod(shape(x))


def __can_be_converted_to_tensor_list_tuple(x: Union[List, Tuple]) -> bool:
    if len(x) == 0:
        return True

    valid_items = all(__can_be_converted_to_tensor_nested(xi) for xi in x)
    if not valid_items:
        return False

    # If all values are scalars-like items
    if all(
        (not isinstance(xi, Sized) or (isinstance(xi, Tensor) and xi.ndim == 0))
        for xi in x
    ):
        return True

    # If all values are sized items with same size
    elif all(isinstance(xi, Sized) for xi in x):
        len0 = len(x[0])
        return all(len(xi) == len0 for xi in x[1:])

    else:
        return False


def __can_be_converted_to_tensor_nested(
    x: Any,
) -> bool:
    if is_builtin_number(x):
        return True
    elif isinstance(x, Tensor) and x.ndim == 0:
        return True
    elif isinstance(x, (np.ndarray, np.generic)) and x.dtype in ACCEPTED_NUMPY_DTYPES:
        return True
    elif isinstance(x, (List, Tuple)):
        return __can_be_converted_to_tensor_list_tuple(x)
    else:
        return False


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
    x: Union[Tensor, np.ndarray, complex]
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
    x: Union[Tensor, np.ndarray, Tuple[float, float]]
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
def prod(
    x: Tensor,
    *,
    dim: Optional[int] = None,
    start: Any = 1,
) -> Tensor:
    ...


@overload
def prod(
    x: Iterable[T_BuiltinNumber],
    *,
    dim: Any = None,
    start: T_BuiltinNumber = 1,
) -> T_BuiltinNumber:
    ...


def prod(
    x: Union[Tensor, Iterable[T_BuiltinNumber]],
    *,
    dim: Optional[int] = None,
    start: T_BuiltinNumber = 1,
) -> Union[Tensor, T_BuiltinNumber]:
    """Returns the product of all elements in input."""
    if isinstance(x, Tensor):
        return torch.prod(x, dim=dim)
    elif isinstance(x, np.ndarray):
        return np.prod(x, axis=dim)
    elif isinstance(x, Iterable):
        if dim is not None:
            msg = f"Invalid argument {dim=}. (expected None with {type(x)=})"
            raise ValueError(msg)
        return builtin_prod(x, start=start)  # type: ignore
    else:
        msg = (
            f"Invalid argument type {type(x)=}. (expected Tensor, ndarray or Iterable)"
        )
        raise TypeError(msg)


def is_floating_point(x: Any) -> TypeGuard[Union[FloatingTensor, np.ndarray, float]]:
    """Returns True if object is a/contains floating-point object(s)."""
    if isinstance(x, Tensor):
        return x.is_floating_point()
    elif isinstance(x, (np.ndarray, np.generic)):
        return numpy_is_floating_point(x)
    else:
        return isinstance(x, float)


def is_complex(x: Any) -> TypeGuard[Union[ComplexFloatingTensor, np.ndarray, complex]]:
    """Returns True if object is a/contains complex-valued object(s)."""
    if isinstance(x, Tensor):
        return x.is_complex()
    elif isinstance(x, (np.ndarray, np.generic)):
        return numpy_is_complex(x)
    else:
        return isinstance(x, complex)


def is_sorted(
    x: Union[Tensor, np.ndarray, Iterable],
    *,
    reverse: bool = False,
    strict: bool = False,
) -> bool:
    """Returns True if the sequence is sorted."""
    if isinstance(x, (Tensor, np.ndarray)):
        if x.ndim != 1:
            msg = f"Invalid number of dims in argument {x.ndim=}. (expected 1)"
            raise ValueError(msg)

        if not reverse and not strict:
            result = (x[:-1] <= x[1:]).all().item()
        elif not reverse and strict:
            result = (x[:-1] < x[1:]).all().item()
        elif reverse and not strict:
            result = (x[:-1] >= x[1:]).all().item()
        else:  # reverse and strict
            result = (x[:-1] > x[1:]).all().item()
        return result  # type: ignore

    elif isinstance(x, Iterable):
        return builtin_is_sorted(x, reverse=reverse, strict=strict)

    else:
        raise TypeError(f"Invalid argument type {type(x)=}.")


@overload
def all_eq(
    x: Union[Tensor, np.ndarray, ScalarLike, Iterable],
    dim: None = None,
) -> bool:
    ...


@overload
def all_eq(
    x: Union[T_TensorLike],
    dim: int,
) -> T_TensorLike:
    ...


def all_eq(
    x: Union[T_TensorLike, ScalarLike, Iterable],
    dim: Union[int, None] = None,
) -> Union[bool, T_TensorLike]:
    """Check if all elements are equal in a tensor, ndarray, iterable or scalar object."""
    if isinstance(x, Tensor):
        if dim is None:
            if x.ndim == 0 or x.nelement() == 0:
                return True
            x = x.reshape(-1)
            return (x[0] == x[1:]).all().item()
        else:
            slices = [slice(None) for _ in range(x.ndim)]
            slices[dim] = 0
            slices.insert(dim + 1, None)
            return (x == x[slices]).all(dim)

    elif isinstance(x, (np.ndarray, np.generic)):
        return numpy_all_eq(x, dim=dim)

    elif dim is not None:
        raise ValueError(f"Invalid argument {dim=} with {type(x)=}.")

    elif is_scalar_like(x):
        return True

    elif isinstance(x, Iterable):
        return builtin_all_eq(x)

    else:
        raise TypeError(f"Invalid argument type {type(x)=}.")


def all_ne(x: Union[Tensor, np.ndarray, ScalarLike, Iterable]) -> bool:
    """Check if all elements are NOT equal in a tensor, ndarray, iterable or scalar object."""
    if isinstance(x, Tensor):
        return len(torch.unique(x)) == x.nelement()
    elif isinstance(x, (np.ndarray, np.generic)):
        return numpy_all_ne(x)
    elif is_scalar_like(x):
        return True
    elif isinstance(x, Iterable):
        return builtin_all_ne(x)
    else:
        raise TypeError(f"Invalid argument type {type(x)=}.")


def average_power(
    x: T_TensorLike,
    dim: Union[int, Tuple[int, ...], None] = -1,
) -> T_TensorLike:
    """Compute average power of a signal along a specified dim/axis."""
    return (abs(x) ** 2).mean(dim)
