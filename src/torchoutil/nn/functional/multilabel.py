#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for conversion between classes indices, multihot, names and probabilities for multilabel classification.
"""

from typing import List, Mapping, Sequence, TypeVar, Union

import torch
from torch import Tensor

from torchoutil.nn.functional.get import get_device

T = TypeVar("T")


def indices_to_multihot(
    indices: Union[Sequence[Sequence[int]], List[Tensor]],
    num_classes: int,
    device: Union[str, torch.device, None] = None,
) -> Tensor:
    """Convert indices of labels to multihot boolean encoding.

    Args:
        indices: List of list of label indices.
        num_classes: Number maximal of unique classes.
        device: PyTorch device of the output tensor.
    """
    device = get_device(device)
    bsize = len(indices)
    multihot = torch.full((bsize, num_classes), False, dtype=torch.bool, device=device)
    for i, indices_i in enumerate(indices):
        if isinstance(indices_i, Tensor):
            indices_i = indices_i.to(device=device)
        else:
            indices_i = torch.as_tensor(indices_i, dtype=torch.long, device=device)
        multihot[i].scatter_(0, indices_i, True)
    return multihot


def indices_to_names(
    indices: Union[Sequence[Sequence[int]], List[Tensor]],
    idx_to_name: Mapping[int, T],
) -> List[List[T]]:
    """Convert indices of labels to names using a mapping.

    Args:
        indices: List of list of label indices.
        idx_to_name: Mapping to convert a class index to its name.
    """
    names = []
    for indices_i in indices:
        names_i = [idx_to_name[idx] for idx in indices_i]  # type: ignore
        names.append(names_i)
    return names


def multihot_to_indices(
    multihot: Union[Tensor, Sequence[Tensor], Sequence[Sequence[bool]]],
) -> List[List[int]]:
    """Convert multihot boolean encoding to indices of labels.

    Args:
        multihot: Multihot labels encoding as 2D matrix.
    """
    preds = []
    for multihot_i in multihot:
        if not isinstance(multihot_i, Tensor):
            multihot_i = torch.as_tensor(multihot_i)
        preds_i = torch.where(multihot_i)[0].tolist()
        preds.append(preds_i)
    return preds


def multihot_to_names(
    multihot: Union[Tensor, Sequence[Tensor], Sequence[Sequence[bool]]],
    idx_to_name: Mapping[int, T],
) -> List[List[T]]:
    """Convert multihot boolean encoding to names using a mapping.

    Args:
        multihot: Multihot labels encoding as 2D matrix.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = multihot_to_indices(multihot)
    names = indices_to_names(indices, idx_to_name)
    return names


def names_to_indices(
    names: List[List[T]],
    idx_to_name: Mapping[int, T],
) -> List[List[int]]:
    """Convert names to indices of labels.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
    """
    name_to_idx = {name: idx for idx, name in idx_to_name.items()}
    indices = []
    for names_i in names:
        indices_i = [name_to_idx[name] for name in names_i]
        indices.append(indices_i)
    return indices


def names_to_multihot(
    names: List[List[T]],
    idx_to_name: Mapping[int, T],
    device: Union[str, torch.device, None] = None,
) -> Tensor:
    """Convert names to multihot boolean encoding.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
        device: PyTorch device of the output tensor.
    """
    indices = names_to_indices(names, idx_to_name)
    multihot = indices_to_multihot(indices, len(idx_to_name), device)
    return multihot


def probs_to_indices(
    probs: Tensor,
    threshold: Union[float, Sequence[float], Tensor],
) -> List[List[int]]:
    """Convert matrix of probabilities to indices of labels.

    Args:
        probs: Output probabilities for each classes.
        threshold: Threshold(s) to binarize probabilities. Can be a scalar or a sequence of (num_classes,) thresholds.
    """
    multihot = probs_to_multihot(probs, threshold)
    preds = multihot_to_indices(multihot)
    return preds


def probs_to_multihot(
    probs: Tensor,
    threshold: Union[float, Sequence[float], Tensor],
) -> Tensor:
    """Convert matrix of probabilities to multihot boolean encoding.

    Args:
        probs: Output probabilities for each classes.
        threshold: Threshold(s) to binarize probabilities. Can be a scalar or a sequence of (num_classes,) thresholds.
    """
    if probs.ndim != 2:
        raise ValueError(
            "Invalid argument probs. (expected a batch of probabilities of shape (N, n_classes))."
        )

    num_classes = probs.shape[-1]
    device = probs.device

    if not isinstance(threshold, Tensor):
        threshold = torch.as_tensor(threshold, dtype=torch.float, device=device)
    else:
        threshold = threshold.to(device=device)

    if threshold.ndim == 0:
        threshold = threshold.repeat(num_classes)
    elif threshold.ndim == 1:
        if threshold.shape[0] != num_classes:
            raise ValueError(
                f"Invalid argument threshold. (number of thresholds is {threshold.shape[0]} but found {num_classes} classes)"
            )
    else:
        raise ValueError(
            f"Invalid number of dimensions in input threshold. (found {threshold.shape=} but expected 0-d or 1-d tensor)"
        )

    multihot = probs >= threshold
    return multihot


def probs_to_names(
    probs: Tensor,
    threshold: Union[float, Sequence[float], Tensor],
    idx_to_name: Mapping[int, T],
) -> List[List[T]]:
    """Convert matrix of probabilities to labels names.

    Args:
        probs: Output probabilities for each classes.
        threshold: Threshold(s) to binarize probabilities. Can be a scalar or a sequence of (num_classes,) thresholds.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = probs_to_indices(probs, threshold)
    names = indices_to_names(indices, idx_to_name)
    return names
