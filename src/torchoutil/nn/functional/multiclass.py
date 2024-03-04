#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for conversion between classes indices, onehot, names and probabilities for multiclass classification.
"""

from typing import List, Mapping, Sequence, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from torchoutil.nn.functional.get import get_device

T = TypeVar("T")


def indices_to_onehot(
    indices: Union[Sequence[int], Tensor],
    num_classes: int,
    device: Union[str, torch.device, None] = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert indices of labels to onehot boolean encoding.

    Args:
        indices: List of list of label indices.
        num_classes: Number maximal of unique classes.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    device = get_device(device)
    indices = torch.as_tensor(indices, device=device, dtype=torch.long)
    onehot = F.one_hot(indices, num_classes)
    onehot = onehot.to(dtype=dtype)
    return onehot


def indices_to_names(
    indices: Union[Sequence[int], Tensor],
    idx_to_name: Mapping[int, T],
) -> List[T]:
    """Convert indices of labels to names using a mapping.

    Args:
        indices: List of list of label indices.
        idx_to_name: Mapping to convert a class index to its name.
    """
    return [idx_to_name[indices_i] for indices_i in indices]  # type: ignore


def onehot_to_indices(
    onehot: Tensor,
) -> List[int]:
    """Convert onehot boolean encoding to indices of labels.

    Args:
        onehot: OneHot labels encoded as 2D matrix.
    """
    return onehot.argmax(dim=-1).tolist()


def onehot_to_names(
    onehot: Tensor,
    idx_to_name: Mapping[int, T],
) -> List[T]:
    """Convert onehot boolean encoding to names using a mapping.

    Args:
        onehot: OneHot labels encoded as 2D matrix.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = onehot_to_indices(onehot)
    names = indices_to_names(indices, idx_to_name)
    return names


def names_to_indices(
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


def names_to_onehot(
    names: List[T],
    idx_to_name: Mapping[int, T],
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
    indices = names_to_indices(names, idx_to_name)
    onehot = indices_to_onehot(indices, len(idx_to_name), device, dtype)
    return onehot


def probs_to_indices(
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
    device: Union[str, torch.device, None] = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert matrix of probabilities to onehot boolean encoding.

    Args:
        probs: Output probabilities for each classes.
    """
    if device is None:
        device = probs.device
    indices = probs_to_indices(probs)
    onehot = indices_to_onehot(indices, probs.shape[-1], device, dtype)
    return onehot


def probs_to_names(
    probs: Tensor,
    idx_to_name: Mapping[int, T],
) -> List[T]:
    """Convert matrix of probabilities to labels names.

    Args:
        probs: Output probabilities for each classes.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = probs_to_indices(probs)
    names = indices_to_names(indices, idx_to_name)
    return names
