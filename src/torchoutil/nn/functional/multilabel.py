#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper functions for conversion between classes indices, multihot, multi-names and probabilities for multilabel classification.
"""

from typing import Hashable, Iterable, List, Mapping, Optional, Sequence, TypeVar, Union

import torch
from torch import Tensor

from torchoutil.core.get import DeviceLike, DTypeLike, get_device, get_dtype
from torchoutil.nn.functional.others import can_be_stacked, to_item
from torchoutil.nn.functional.pad import pad_and_stack_rec
from torchoutil.nn.functional.transform import to_tensor
from torchoutil.pyoutil.typing import is_sequence_int
from torchoutil.types import LongTensor, is_number_like, is_tensor_like
from torchoutil.types._typing import TensorLike

T_Name = TypeVar("T_Name", bound=Hashable)


def indices_to_multihot(
    indices: Union[Sequence[Union[Sequence[int], TensorLike]], TensorLike],
    num_classes: int,
    *,
    padding_idx: Optional[int] = None,
    device: DeviceLike = None,
    dtype: DTypeLike = torch.bool,
) -> Tensor:
    """Convert indices of labels to multihot boolean encoding for **multilabel** classification.

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
    dtype = get_dtype(dtype)

    def _impl(x) -> Tensor:
        if is_tensor_like(x) and not _is_valid_indices(x):
            raise ValueError(f"Invalid argument shape {x.shape=}.")

        if (is_tensor_like(x) and x.ndim == 1) or is_sequence_int(x):
            x = torch.as_tensor(x, dtype=torch.long, device=device)

            if padding_idx is not None:
                x = torch.where(x == padding_idx, num_classes, x)
                target_num_classes = num_classes + 1
            else:
                target_num_classes = num_classes

            multihot = torch.full(
                (target_num_classes,),
                False,
                dtype=dtype,
                device=device,
            )
            multihot.scatter_(0, x, True)

            if padding_idx is not None:
                multihot = multihot[..., :-1]

            return multihot

        elif isinstance(x, Iterable):
            result = [_impl(xi) for xi in x]
            return torch.stack(result)

        else:
            raise ValueError(f"Invalid argument {x=}.")

    multihot = _impl(indices)
    return multihot


def indices_to_multinames(
    indices: Union[Sequence[Union[Sequence[int], TensorLike]], TensorLike],
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    *,
    padding_idx: Optional[int] = None,
) -> List[List[T_Name]]:
    """Convert indices of labels to names using a mapping for **multilabel** classification.

    Args:
        indices: List of list of label indices.
        idx_to_name: Mapping to convert a class index to its name.
        padding_idx: Optional pad value to ignore.
    """
    if not isinstance(indices, Iterable):
        raise TypeError(f"Invalid argument {indices=}. (not an iterable)")

    def _impl(x) -> Union[int, list]:
        if is_number_like(x):
            return idx_to_name[to_item(x)]  # type: ignore
        elif isinstance(x, Iterable):
            return [
                _impl(xi)
                for xi in x
                if padding_idx is None or not is_number_like(xi) or xi != padding_idx
            ]
        else:
            msg = f"Invalid argument {x=}. (not present in idx_to_name and not an iterable type)"
            raise ValueError(msg)

    names = _impl(indices)
    return names  # type: ignore


def multihot_to_indices(
    multihot: Union[
        TensorLike, Sequence[TensorLike], Sequence[Sequence[bool]], Sequence
    ],
    *,
    keep_tensor: bool = False,
    padding_idx: Optional[int] = None,
    dim: int = -1,
) -> Union[List, LongTensor]:
    """Convert multihot boolean encoding to indices of labels for **multilabel** classification.

    Args:
        multihot: Multihot labels encoded as 2D matrix. Must be convertible to Tensor.
        keep_tensor: If True, output will be converted to a tensor if possible. defaults to False.
        padding_idx: Class index fill value. When none, output will not be padded. defaults to None.
        dim: Dimension of classes. defaults to -1.
    """
    multihot = to_tensor(multihot)
    multihot = multihot.transpose(dim, -1)

    if not _is_valid_indices(multihot):
        msg = f"Invalid argument shape {multihot=}. (expected first axis should be > 0)"
        raise ValueError(msg)

    def _impl(x: TensorLike) -> Union[list, LongTensor]:
        if x.ndim == 1:
            x = torch.as_tensor(x, dtype=torch.bool)
            preds = torch.where(x)[0]
            if not keep_tensor:
                preds = preds.tolist()
            return preds  # type: ignore

        elif x.ndim > 1:
            preds = [_impl(multihot_i) for multihot_i in x]
            if padding_idx is not None:
                preds = pad_and_stack_rec(preds, padding_idx, dtype=torch.long)
                if not keep_tensor:
                    preds = preds.tolist()
            elif keep_tensor and can_be_stacked(preds):
                preds = torch.stack(preds)
            return preds  # type: ignore

        else:
            msg = f"Invalid argument {x=}. (found {x.ndim} dims)"
            raise ValueError(msg)

    result = _impl(multihot)
    return result


def multihot_to_multinames(
    multihot: Union[TensorLike, Sequence[TensorLike], Sequence[Sequence[bool]]],
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    *,
    dim: int = -1,
) -> List[List[T_Name]]:
    """Convert multihot boolean encoding to names using a mapping for **multilabel** classification.

    Args:
        multihot: Multihot labels encoded as 2D matrix.
        idx_to_name: Mapping to convert a class index to its name.
        dim: Dimension of classes. defaults to -1.
    """
    indices = multihot_to_indices(multihot, dim=dim)
    names = indices_to_multinames(indices, idx_to_name)
    return names


def multinames_to_indices(
    names: List[List[T_Name]],
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
) -> List[List[int]]:
    """Convert names to indices of labels for **multilabel** classification.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
    """
    if isinstance(idx_to_name, Mapping):
        name_to_idx = {name: idx for idx, name in idx_to_name.items()}
    else:
        name_to_idx = {name: idx for idx, name in enumerate(idx_to_name)}
    del idx_to_name

    def _impl(x) -> Union[int, list]:
        if isinstance(x, Hashable) and x in name_to_idx:
            return name_to_idx[x]  # type: ignore
        elif isinstance(x, Iterable):
            return [_impl(xi) for xi in x]
        else:
            msg = f"Invalid argument {x=}. (not present in idx_to_name and not an iterable type)"
            raise ValueError(msg)

    indices = _impl(names)
    return indices  # type: ignore


def multinames_to_multihot(
    names: List[List[T_Name]],
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
    *,
    device: DeviceLike = None,
    dtype: DTypeLike = torch.bool,
) -> Tensor:
    """Convert names to multihot boolean encoding for **multilabel** classification.

    Args:
        names: List of list of label names.
        idx_to_name: Mapping to convert a class index to its name.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    indices = multinames_to_indices(names, idx_to_name)
    multihot = indices_to_multihot(
        indices,
        len(idx_to_name),
        device=device,
        dtype=dtype,
    )
    return multihot


def probs_to_indices(
    probs: TensorLike,
    threshold: Union[float, Sequence[float], TensorLike],
    *,
    padding_idx: Optional[int] = None,
    dim: int = -1,
) -> Union[List, LongTensor]:
    """Convert matrix of probabilities to indices of labels for **multilabel** classification.

    Args:
        probs: Output probabilities for each classes.
        threshold: Threshold(s) to binarize probabilities. Can be a scalar or a sequence of (num_classes,) thresholds.
        padding_idx: Class index fill value. When none, output will not be padded. defaults to None.
        dim: Dimension of classes. defaults to -1.
    """
    multihot = probs_to_multihot(probs, threshold, dim=dim)
    indices = multihot_to_indices(multihot, padding_idx=padding_idx, dim=dim)
    return indices


def probs_to_multihot(
    probs: TensorLike,
    threshold: Union[float, Sequence[float], TensorLike],
    *,
    dim: int = -1,
    device: DeviceLike = None,
    dtype: DTypeLike = torch.bool,
) -> Tensor:
    """Convert matrix of probabilities to multihot boolean encoding for **multilabel** classification.

    Args:
        probs: Output probabilities for each class.
        threshold: Threshold(s) to binarize probabilities. Can be a scalar or a sequence of (num_classes,) thresholds.
        dim: Dimension of classes. defaults to -1.
        device: PyTorch device of the output tensor.
        dtype: PyTorch DType of the output tensor.
    """
    probs = to_tensor(probs)
    if probs.ndim == 0:
        msg = f"Invalid argument ndim {probs.ndim=}. (expected at least 1 dim)."
        raise ValueError(msg)

    num_classes = probs.shape[dim]
    if device is None:
        device = probs.device
    else:
        device = get_device(device)
    dtype = get_dtype(dtype)

    if not isinstance(threshold, Tensor):
        threshold = torch.as_tensor(threshold, dtype=probs.dtype, device=device)
    else:
        threshold = threshold.to(device=device)

    if threshold.ndim == 0:
        threshold = threshold.repeat(num_classes)
    elif threshold.ndim == 1:
        if threshold.shape[0] != num_classes:
            msg = f"Invalid argument threshold. (number of thresholds is {threshold.shape[0]} but found {num_classes} classes)"
            raise ValueError(msg)
    else:
        msg = f"Invalid number of dimensions in input threshold. (found {threshold.shape=} but expected 0-d or 1-d tensor)"
        raise ValueError(msg)

    dim = dim % probs.ndim
    slices = [(slice(None) if i == dim else None) for i in range(probs.ndim)]
    threshold = threshold[slices]

    multihot = probs >= threshold
    multihot = multihot.to(dtype=dtype)
    return multihot


def probs_to_multinames(
    probs: TensorLike,
    threshold: Union[float, Sequence[float], TensorLike],
    idx_to_name: Union[Mapping[int, T_Name], Sequence[T_Name]],
) -> List[List[T_Name]]:
    """Convert matrix of probabilities to labels names for **multilabel** classification.

    Args:
        probs: Output probabilities for each classes.
        threshold: Threshold(s) to binarize probabilities. Can be a scalar or a sequence of (num_classes,) thresholds.
        idx_to_name: Mapping to convert a class index to its name.
    """
    indices = probs_to_indices(probs, threshold)
    names = indices_to_multinames(indices, idx_to_name)
    return names


def _is_valid_indices(x: TensorLike, dim: int = -1) -> bool:
    """Returns True if tensor has valid shape for indices_to_multihot."""
    shape = list(x.shape)
    dim = dim % len(shape)
    try:
        idx = shape.index(0)
        return idx == dim
    except ValueError:
        return True
