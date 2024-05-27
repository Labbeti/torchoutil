#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for conversion between classes indices, onehot, names and probabilities for multiclass classification.
"""

from typing import Hashable, Iterable, List, Mapping, Optional, Sequence, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.types import Device

from torchoutil.nn.functional.get import get_device
from torchoutil.utils.type_checks import is_scalar

T = TypeVar("T")


def index_to_onehot(
    index: Union[Sequence[int], Tensor, Sequence],
    num_classes: int,
    *,
    padding_idx: Optional[int] = None,
    device: Device = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert indices of labels to onehot boolean encoding.

    Args:
        indices: List label indices.
            Can be a nested list of indices, but it should be convertible to Tensor.
        num_classes: Number maximal of unique classes.
        padding_idx: Class index to ignore. Output will contains only zeroes for this value. defaults to None.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    device = get_device(device)
    index = torch.as_tensor(index, device=device, dtype=torch.long)

    if padding_idx is not None:
        mask = index == padding_idx
        index = torch.where(mask, num_classes, index)
        num_classes += 1

    onehot: Tensor = F.one_hot(index, num_classes)
    onehot = onehot.to(dtype=dtype)

    if padding_idx is not None:
        onehot = onehot[..., :-1]

    return onehot


def index_to_name(
    index: Union[Sequence[int], Tensor, Sequence],
    idx_to_name: Union[Mapping[int, T], Sequence[T]],
) -> List[T]:
    """Convert indices of labels to names using a mapping.

    Args:
        indices: List of list of label indices.
        idx_to_name: Mapping to convert a class index to its name.
    """

    def index_to_name_impl(x) -> Union[T, list]:
        if is_scalar(x):
            return idx_to_name[x]  # type: ignore
        elif isinstance(x, Iterable):
            return [index_to_name_impl(xi) for xi in x]
        else:
            raise ValueError(
                f"Invalid argument {x=}. (not present in idx_to_name and not an iterable type)"
            )

    name = index_to_name_impl(index)
    return name  # type: ignore


def onehot_to_index(
    onehot: Tensor,
    *,
    dim: int = -1,
) -> List[int]:
    """Convert onehot boolean encoding to indices of labels.

    Args:
        onehot: Onehot labels encoded as 2D matrix.
    """
    onehot = onehot.int()
    index = onehot.argmax(dim=dim)
    index = index.tolist()
    return index


def onehot_to_name(
    onehot: Tensor,
    idx_to_name: Union[Mapping[int, T], Sequence[T]],
    *,
    dim: int = -1,
) -> List[T]:
    """Convert onehot boolean encoding to names using a mapping.

    Args:
        onehot: Onehot labels encoded as 2D matrix.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = onehot_to_index(onehot, dim=dim)
    names = index_to_name(indices, idx_to_name)
    return names


def name_to_index(
    name: List[T],
    idx_to_name: Union[Mapping[int, T], Sequence[T]],
) -> List[int]:
    """Convert names to indices of labels.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
    """
    if isinstance(idx_to_name, Mapping):
        name_to_idx = {name: idx for idx, name in idx_to_name.items()}
    else:
        name_to_idx = {name: idx for idx, name in enumerate(idx_to_name)}
    del idx_to_name

    def name_to_index_impl(x) -> Union[T, list]:
        if isinstance(x, Hashable) and x in name_to_idx:
            return name_to_idx[x]  # type: ignore
        elif isinstance(x, Iterable):
            return [name_to_index_impl(xi) for xi in x]
        else:
            raise ValueError(
                f"Invalid argument {x=}. (not present in name_to_idx and not an iterable type)"
            )

    index = name_to_index_impl(name)
    return index  # type: ignore


def name_to_onehot(
    name: List[T],
    idx_to_name: Union[Mapping[int, T], Sequence[T]],
    *,
    device: Device = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert names to onehot boolean encoding.

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
) -> List[int]:
    """Convert matrix of probabilities to indices of labels.

    Args:
        probs: Output probabilities for each classes.
    """
    index = probs.argmax(dim=dim)
    index = index.tolist()
    return index


def probs_to_onehot(
    probs: Tensor,
    *,
    dim: int = -1,
    device: Device = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert matrix of probabilities to onehot boolean encoding.

    Args:
        probs: Output probabilities for each classes.
    """
    if device is None:
        device = probs.device
    indices = probs_to_index(probs, dim=dim)
    onehot = index_to_onehot(indices, probs.shape[-1], device=device, dtype=dtype)
    return onehot


def probs_to_name(
    probs: Tensor,
    idx_to_name: Union[Mapping[int, T], Sequence[T]],
    *,
    dim: int = -1,
) -> List[T]:
    """Convert matrix of probabilities to labels names.

    Args:
        probs: Output probabilities for each classes.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = probs_to_index(probs, dim=dim)
    names = index_to_name(indices, idx_to_name)
    return names
