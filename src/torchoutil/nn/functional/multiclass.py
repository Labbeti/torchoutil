#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for conversion between classes indices, onehot, names and probabilities for multiclass classification.
"""

from typing import List, Mapping, Optional, Sequence, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from torchoutil.nn.functional.get import get_device

T = TypeVar("T")


def index_to_onehot(
    indices: Union[Sequence[int], Tensor, Sequence],
    num_classes: int,
    *,
    padding_idx: Optional[int] = None,
    device: Union[str, torch.device, None] = None,
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
    indices = torch.as_tensor(indices, device=device, dtype=torch.long)

    if padding_idx is not None:
        mask = indices == padding_idx
        indices = torch.where(mask, num_classes, indices)
        num_classes += 1

    onehot: Tensor = F.one_hot(indices, num_classes)
    onehot = onehot.to(dtype=dtype)

    if padding_idx is not None:
        onehot = onehot[..., :-1]

    return onehot


def index_to_name(
    indices: Union[Sequence[int], Tensor],
    idx_to_name: Mapping[int, T],
) -> List[T]:
    """Convert indices of labels to names using a mapping.

    Args:
        indices: List of list of label indices.
        idx_to_name: Mapping to convert a class index to its name.
    """
    return [idx_to_name[indices_i] for indices_i in indices]  # type: ignore


def onehot_to_index(
    onehot: Tensor,
) -> List[int]:
    """Convert onehot boolean encoding to indices of labels.

    Args:
        onehot: Onehot labels encoded as 2D matrix.
    """
    return onehot.int().argmax(dim=-1).tolist()


def onehot_to_name(
    onehot: Tensor,
    idx_to_name: Mapping[int, T],
) -> List[T]:
    """Convert onehot boolean encoding to names using a mapping.

    Args:
        onehot: Onehot labels encoded as 2D matrix.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = onehot_to_index(onehot)
    names = index_to_name(indices, idx_to_name)
    return names


def name_to_index(
    names: List[T],
    idx_to_name: Mapping[int, T],
) -> List[int]:
    """Convert names to indices of labels.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
    """
    name_to_idx = {name: idx for idx, name in idx_to_name.items()}
    indices = [name_to_idx[name] for name in names]
    return indices


def name_to_onehot(
    names: List[T],
    idx_to_name: Mapping[int, T],
    *,
    device: Union[str, torch.device, None] = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert names to onehot boolean encoding.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    indices = name_to_index(names, idx_to_name)
    onehot = index_to_onehot(indices, len(idx_to_name), device=device, dtype=dtype)
    return onehot


def probs_to_index(
    probs: Tensor,
) -> List[int]:
    """Convert matrix of probabilities to indices of labels.

    Args:
        probs: Output probabilities for each classes.
    """
    indices = probs.argmax(dim=-1)
    indices = indices.tolist()
    return indices


def probs_to_onehot(
    probs: Tensor,
    *,
    device: Union[str, torch.device, None] = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert matrix of probabilities to onehot boolean encoding.

    Args:
        probs: Output probabilities for each classes.
    """
    if device is None:
        device = probs.device
    indices = probs_to_index(probs)
    onehot = index_to_onehot(indices, probs.shape[-1], device=device, dtype=dtype)
    return onehot


def probs_to_name(
    probs: Tensor,
    idx_to_name: Mapping[int, T],
) -> List[T]:
    """Convert matrix of probabilities to labels names.

    Args:
        probs: Output probabilities for each classes.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = probs_to_index(probs)
    names = index_to_name(indices, idx_to_name)
    return names
