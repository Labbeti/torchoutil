#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Union,
)

import torch

from torch import Tensor

from extentorch.nn.functional.get import get_device, _DEVICE_CUDA_IF_AVAILABLE


def get_inverse_perm(indexes: Tensor, dim: int = -1) -> Tensor:
    """Return inverse permutation indexes.
    The output will be a tensor of shape (..., N).

    Args:
        indexes: Original permutation indexes as tensor of shape (..., N).
        dim: Dimension of indexes. defaults to -1.
    """
    arange = torch.arange(
        indexes.shape[dim],
        dtype=indexes.dtype,
        device=indexes.device,
    )
    arange = arange.expand(*indexes.shape)
    indexes_inv = torch.empty_like(indexes)
    indexes_inv = indexes_inv.scatter(dim, indexes, arange)
    return indexes_inv


def randperm_diff(
    size: int,
    generator: Union[None, int, torch.Generator] = None,
    device: Union[str, torch.device, None] = _DEVICE_CUDA_IF_AVAILABLE,
) -> Tensor:
    """This function ensure that every value i cannot be the element at index i.
    The output will be a tensor of shape (size,).

    Args:
        size: The number of indexes. Cannot be < 2.
        seed: The seed or torch.Generator used to generate permutation.
        device: The PyTorch device of the output indexes tensor.

    Example 1
    ----------
        >>> torch.randperm(5)
        tensor([1, 4, 2, 5, 0])  # 2 is the element of index 2 !
        >>> randperm_diff(5)
        tensor([2, 0, 4, 1, 3])
    """
    if size < 2:
        raise ValueError(f"Invalid argument {size=} < 2 for randperm_diff.")

    device = get_device(device)
    if isinstance(generator, int):
        generator = torch.Generator().manual_seed(generator)

    perm_kws: dict[str, Any] = dict(generator=generator, device=device)
    arange = torch.arange(size, device=device)
    perm = torch.randperm(size, **perm_kws)

    while perm.eq(arange).any():
        perm = torch.randperm(size, **perm_kws)
    return perm


def get_perm(t1: Tensor, t2: Tensor) -> Tensor:
    """Find permutation between two vectors t1 and t2 which contains values from 0 to N-1.

    Example 1::
    ----------
        >>> t1 = torch.as_tensor([0, 1, 2, 4, 3, 6, 5, 7])
        >>> t2 = torch.as_tensor([0, 2, 1, 4, 3, 5, 6, 7])
        >>> indexes = get_permutation(t1, t2)
        >>> (t1[indexes] == t2).all().item()
        True
    """
    i1 = (t1[..., None, :] == t2[..., :, None]).int().argmax(dim=-2)
    return i1


def insert_at_indices(
    x: Tensor,
    indexes: Tensor,
    values: Union[float, int, Tensor],
) -> Tensor:
    """Insert value(s) in vector at specified indexes.

    Example 1::
    ----------
        >>> x = torch.as_tensor([1, 1, 2, 2, 2, 3])
        >>> indexes = torch.as_tensor([2, 5])
        >>> values = 4
        >>> insert_values(x, indexes, values)
        tensor([1, 1, 4, 2, 2, 2, 4, 3])
    """
    out = torch.empty((x.shape[0] + indexes.shape[0]), dtype=x.dtype, device=x.device)
    indexes = indexes + torch.arange(indexes.shape[0], device=indexes.device)
    out[indexes] = values
    mask = torch.full((out.shape[0],), True, dtype=torch.bool)
    mask[indexes] = False
    out[mask] = x
    return out


def remove_at_indices(
    x: Tensor,
    indexes: Tensor,
) -> Tensor:
    """Remove value(s) in vector at specified indexes."""
    indexes = indexes + torch.arange(indexes.shape[0], device=indexes.device)
    mask = torch.full((x.shape[0],), True, dtype=torch.bool)
    mask[indexes] = False
    out = x[mask]
    return out
