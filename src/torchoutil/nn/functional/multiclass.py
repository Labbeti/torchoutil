#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for conversion between classes indices, onehot, names and probabilities for multiclass classification.
"""

from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import torch
from torch import Tensor
from torch.nn import functional as F

from torchoutil.core.get import DeviceLike, DTypeLike, get_device, get_dtype
from torchoutil.extras.numpy import np
from torchoutil.nn.functional.others import nelement, to_item
from torchoutil.pyoutil.logging import warn_once
from torchoutil.types import LongTensor, is_number_like
from torchoutil.types._typing import TensorLike

T_Name = TypeVar("T_Name", bound=Hashable)


def index_to_onehot(
    index: Union[Sequence[int], TensorLike, Sequence],
    num_classes: int,
    *,
    padding_idx: Optional[int] = None,
    device: DeviceLike = None,
    dtype: DTypeLike = torch.bool,
) -> Tensor:
    """Convert indices of labels to onehot boolean encoding for **multiclass** classification.

    Args:
        indices: List label indices.
            Can be a nested list of indices, but it should be convertible to Tensor.
        num_classes: Number maximal of unique classes.
        padding_idx: Class index to ignore. Output will contains only zeroes for this value. defaults to None.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    device = get_device(device)
    dtype = get_dtype(dtype)
    index = torch.as_tensor(index, device=device, dtype=torch.long)

    if padding_idx is not None:
        mask = index == padding_idx
        index = torch.where(mask, num_classes, index)
        num_classes += 1

    if index.nelement() > 0 and not (0 <= index.min() <= index.max() < num_classes):
        msg = f"Invalid argument {index=}. (expected 0 <= min={index.min()} <= max={index.max()} < {num_classes=})"
        raise ValueError(msg)

    onehot: Tensor = F.one_hot(index, num_classes)
    onehot = onehot.to(dtype=dtype)

    if padding_idx is not None:
        onehot = onehot[..., :-1]

    return onehot


def index_to_name(
    index: Union[Sequence[int], TensorLike, Sequence],
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    *,
    is_number_fn: Callable[[Any], bool] = is_number_like,
) -> List[T_Name]:
    """Convert indices of labels to names using a mapping for **multiclass** classification.

    Args:
        indices: List of list of label indices.
        idx_to_name: Mapping to convert a class index to its name.
        is_number_fn: Type guard to check if a value is a scalar number. defaults to `is_number_like`.
    """
    if (
        isinstance(index, (Tensor, np.ndarray))
        and nelement(index) == 0
        and index.ndim > 1
    ):
        msg = f"Found 0 elements in {index=} but {index.ndim=} > 1, which means that we will lose information about shape when converting to names."
        warn_once(msg, __name__)

    def _impl(x) -> Union[T_Name, list]:
        if is_number_fn(x):
            return idx_to_name[to_item(x)]  # type: ignore
        elif isinstance(x, Iterable):
            return [_impl(xi) for xi in x]
        else:
            msg = f"Invalid argument {x=}. (not present in idx_to_name and not an iterable type)"
            raise ValueError(msg)

    name = _impl(index)
    return name  # type: ignore


def onehot_to_index(
    onehot: Tensor,
    *,
    padding_idx: Optional[int] = None,
    dim: int = -1,
) -> LongTensor:
    """Convert onehot boolean encoding to indices of labels for **multiclass** classification.

    Args:
        onehot: Onehot labels encoded as 2D matrix.
        padding_idx: Class index placeholder when input contains only zeroes. defaults to None.
        dim: Dimension of classes. defaults to -1.
    """
    onehot = onehot.int()
    index = onehot.argmax(dim=dim)

    if padding_idx is not None:
        empty = onehot.eq(False).all(dim=dim)
        index = torch.where(empty, padding_idx, index)

    return index  # type: ignore


def onehot_to_name(
    onehot: Tensor,
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    *,
    dim: int = -1,
) -> List[T_Name]:
    """Convert onehot boolean encoding to names using a mapping for **multiclass** classification.

    Args:
        onehot: Onehot labels encoded as 2D matrix.
        idx_to_name: Mapping to convert a class index to its name.
        dim: Dimension of classes. defaults to -1.
    """
    indices = onehot_to_index(onehot, dim=dim)
    names = index_to_name(indices, idx_to_name)
    return names


def name_to_index(
    name: List[T_Name],
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
) -> Tensor:
    """Convert names to indices of labels for **multiclass** classification.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
    """
    if isinstance(idx_to_name, Mapping):
        name_to_idx = {name: idx for idx, name in idx_to_name.items()}
    else:
        name_to_idx = {name: idx for idx, name in enumerate(idx_to_name)}
    del idx_to_name

    def _impl(x) -> Union[T_Name, list]:
        if isinstance(x, Hashable) and x in name_to_idx:
            return name_to_idx[x]  # type: ignore
        elif isinstance(x, Iterable):
            return [_impl(xi) for xi in x]
        else:
            msg = f"Invalid argument {x=}. (not present in name_to_idx and not an iterable type)"
            raise ValueError(msg)

    index = _impl(name)
    index = torch.as_tensor(index, dtype=torch.long)
    return index  # type: ignore


def name_to_onehot(
    name: List[T_Name],
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    *,
    device: DeviceLike = None,
    dtype: DTypeLike = torch.bool,
) -> Tensor:
    """Convert names to onehot boolean encoding for **multiclass** classification.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    index = name_to_index(name, idx_to_name)
    onehot = index_to_onehot(index, len(idx_to_name), device=device, dtype=dtype)
    return onehot


def probs_to_index(
    probs: Tensor,
    *,
    dim: int = -1,
) -> LongTensor:
    """Convert matrix of probabilities to indices of labels for **multiclass** classification.

    Args:
        probs: Output probabilities for each classes.
        dim: Dimension of classes. defaults to -1.
    """
    index = probs.argmax(dim=dim)
    return index  # type: ignore


def probs_to_onehot(
    probs: Tensor,
    *,
    dim: int = -1,
    device: DeviceLike = None,
    dtype: DTypeLike = torch.bool,
) -> Tensor:
    """Convert matrix of probabilities to onehot boolean encoding for **multiclass** classification.

    Args:
        probs: Output probabilities for each classes.
        dim: Dimension of classes. defaults to -1.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    if device is None:
        device = probs.device
    indices = probs_to_index(probs, dim=dim)
    onehot = index_to_onehot(indices, probs.shape[-1], device=device, dtype=dtype)
    return onehot


def probs_to_name(
    probs: Tensor,
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    *,
    dim: int = -1,
) -> List[T_Name]:
    """Convert matrix of probabilities to labels names for **multiclass** classification.

    Args:
        probs: Output probabilities for each classes.
        idx_to_name: Mapping to convert a class index to its name.
        dim: Dimension of classes. defaults to -1.
    """
    indices = probs_to_index(probs, dim=dim)
    names = index_to_name(indices, idx_to_name)
    return names
