#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for conversion between classes indices, multihot, names and probabilities for multilabel classification.
"""

from typing import Hashable, Iterable, List, Mapping, Optional, Sequence, TypeVar, Union

import torch
from torch import Tensor
from torch.types import Device

from torchoutil.nn.functional.get import get_device
from torchoutil.nn.functional.pad import pad_and_stack_rec
from torchoutil.utils.type_checks import is_scalar, is_sequence_bool, is_sequence_int

T = TypeVar("T")


def indices_to_multihot(
    indices: Union[Sequence[Union[Sequence[int], Tensor]], Tensor],
    num_classes: int,
    *,
    padding_idx: Optional[int] = None,
    device: Device = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert indices of labels to multihot boolean encoding.

    Args:
        indices: List of list of label indices. Values should be integers in range [0..num_classes-1]
        num_classes: Number maximal of unique classes.
        padding_idx: Optional pad value to ignore.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    if device is None and isinstance(indices, Tensor):
        device = indices.device
    else:
        device = get_device(device)

    def _indices_to_multihot_impl(x) -> Tensor:
        if isinstance(x, Tensor) and not _is_valid_indices(x):
            raise ValueError(f"Invalid argument shape {x.shape=}.")

        if (isinstance(x, Tensor) and x.ndim == 1) or is_sequence_int(x):
            x = torch.as_tensor(x, dtype=torch.long, device=device)

            if padding_idx is not None:
                x = torch.where(x == padding_idx, num_classes, x)
                target_num_classes = num_classes + 1
            else:
                target_num_classes = num_classes

            multihot = torch.full(
                (target_num_classes,), False, dtype=dtype, device=device
            )
            multihot.scatter_(0, x, True)

            if padding_idx is not None:
                multihot = multihot[..., :-1]

            return multihot

        elif isinstance(x, Iterable):
            result = [_indices_to_multihot_impl(xi) for xi in x]
            return torch.stack(result)

        else:
            raise ValueError(f"Invalid argument {x=}.")

    multihot = _indices_to_multihot_impl(indices)
    return multihot


def indices_to_names(
    indices: Union[Sequence[Union[Sequence[int], Tensor]], Tensor],
    idx_to_name: Mapping[int, T],
    *,
    padding_idx: Optional[int] = None,
) -> List[List[T]]:
    """Convert indices of labels to names using a mapping.

    Args:
        indices: List of list of label indices.
        idx_to_name: Mapping to convert a class index to its name.
        padding_idx: Optional pad value to ignore.
    """

    def _indices_to_names_impl(x) -> Union[int, list]:
        if is_scalar(x):
            return idx_to_name[x]  # type: ignore
        elif isinstance(x, Iterable):
            return [
                _indices_to_names_impl(xi)
                for xi in x
                if padding_idx is None or not is_scalar(xi) or xi != padding_idx
            ]
        else:
            raise ValueError(
                f"Invalid argument {x=}. (not present in idx_to_name and not an iterable type)"
            )

    if not isinstance(indices, Iterable):
        raise TypeError(f"Invalid argument {indices=}. (not an iterable)")

    names = _indices_to_names_impl(indices)
    return names  # type: ignore


def multihot_to_indices(
    multihot: Union[Tensor, Sequence[Tensor], Sequence[Sequence[bool]]],
    *,
    padding_idx: Optional[int] = None,
) -> List[List[int]]:
    """Convert multihot boolean encoding to indices of labels.

    Args:
        multihot: Multihot labels encoded as 2D matrix.
    """
    if is_scalar(multihot) or (
        isinstance(multihot, Tensor) and not _is_valid_indices(multihot)
    ):
        raise ValueError(
            f"Invalid argument shape {multihot=}. (expected at least 1 dimension and the first axis should be > 0)"
        )

    def _multihot_to_indices_impl(
        x: Union[Tensor, Sequence],
    ) -> list:
        if (isinstance(x, Tensor) and x.ndim == 1) or is_sequence_bool(x):
            x = torch.as_tensor(x, dtype=torch.bool)
            preds = torch.where(x)[0].tolist()
            return preds
        elif (isinstance(x, Tensor) and x.ndim > 1) or isinstance(x, Sequence):
            preds = [_multihot_to_indices_impl(multihot_i) for multihot_i in x]
            if padding_idx is not None:
                preds = pad_and_stack_rec(preds, padding_idx)
                preds = preds.tolist()
            return preds

        else:
            raise ValueError(
                f"Invalid argument {x=}. (not present in idx_to_name and not an iterable type)"
            )

    return _multihot_to_indices_impl(multihot)


def multihot_to_names(
    multihot: Union[Tensor, Sequence[Tensor], Sequence[Sequence[bool]]],
    idx_to_name: Mapping[int, T],
) -> List[List[T]]:
    """Convert multihot boolean encoding to names using a mapping.

    Args:
        multihot: Multihot labels encoded as 2D matrix.
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
    del idx_to_name

    def _names_to_indices_impl(x) -> Union[int, list]:
        if isinstance(x, Hashable) and x in name_to_idx:
            return name_to_idx[x]  # type: ignore
        elif isinstance(x, Iterable):
            return [_names_to_indices_impl(xi) for xi in x]
        else:
            raise ValueError(
                f"Invalid argument {x=}. (not present in idx_to_name and not an iterable type)"
            )

    indices = _names_to_indices_impl(names)
    return indices  # type: ignore


def names_to_multihot(
    names: List[List[T]],
    idx_to_name: Mapping[int, T],
    *,
    device: Device = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert names to multihot boolean encoding.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    indices = names_to_indices(names, idx_to_name)
    multihot = indices_to_multihot(
        indices,
        len(idx_to_name),
        device=device,
        dtype=dtype,
    )
    return multihot


def probs_to_indices(
    probs: Tensor,
    threshold: Union[float, Sequence[float], Tensor],
    *,
    padding_idx: Optional[int] = None,
) -> List[List[int]]:
    """Convert matrix of probabilities to indices of labels.

    Args:
        probs: Output probabilities for each classes.
        threshold: Threshold(s) to binarize probabilities. Can be a scalar or a sequence of (num_classes,) thresholds.
    """
    multihot = probs_to_multihot(probs, threshold)
    indices = multihot_to_indices(multihot, padding_idx=padding_idx)
    return indices


def probs_to_multihot(
    probs: Tensor,
    threshold: Union[float, Sequence[float], Tensor],
    *,
    device: Device = None,
    dtype: Union[torch.dtype, None] = torch.bool,
) -> Tensor:
    """Convert matrix of probabilities to multihot boolean encoding.

    Args:
        probs: Output probabilities for each classes.
        threshold: Threshold(s) to binarize probabilities. Can be a scalar or a sequence of (num_classes,) thresholds.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    if probs.ndim != 2:
        raise ValueError(
            "Invalid argument probs. (expected a batch of probabilities of shape (N, n_classes))."
        )

    num_classes = probs.shape[-1]
    if device is None:
        device = probs.device
    else:
        device = get_device(device)

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
    multihot = multihot.to(dtype=dtype)
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


def _is_valid_indices(x: Tensor, dim: int = -1) -> bool:
    """Returns True if tensor has valid shape for indices_to_multihot."""
    shape = list(x.shape)
    dim = dim % len(shape)
    try:
        idx = shape.index(0)
        return idx == dim
    except ValueError:
        return True
