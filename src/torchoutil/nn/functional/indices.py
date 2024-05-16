#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from torch.types import Device, Number

from torchoutil.nn.functional.get import _DEVICE_CUDA_IF_AVAILABLE, get_device


def get_inverse_perm(indices: Tensor, dim: int = -1) -> Tensor:
    """Return inverse permutation indices.
    The output will be a tensor of shape (..., N).

    Args:
        indices: Original permutation indices as tensor of shape (..., N).
        dim: Dimension of indices. defaults to -1.
    """
    arange = torch.arange(
        indices.shape[dim],
        dtype=indices.dtype,
        device=indices.device,
    )
    arange = arange.expand(*indices.shape)
    indices_inv = torch.empty_like(indices)
    indices_inv = indices_inv.scatter(dim, indices, arange)
    return indices_inv


def randperm_diff(
    size: int,
    generator: Union[None, int, torch.Generator] = None,
    device: Device = _DEVICE_CUDA_IF_AVAILABLE,
) -> Tensor:
    """This function ensure that every value i cannot be the element at index i.
    The output will be a tensor of shape (size,).

    Args:
        size: The number of indices. Cannot be < 2.
        seed: The seed or torch.Generator used to generate permutation.
        device: The PyTorch device of the output indices tensor.

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

    perm_kws: Dict[str, Any] = dict(generator=generator, device=device)
    arange = torch.arange(size, device=device)
    perm = torch.randperm(size, **perm_kws)

    while perm.eq(arange).any():
        perm = torch.randperm(size, **perm_kws)
    return perm


def get_perm_indices(t1: Tensor, t2: Tensor) -> Tensor:
    """Find permutation between two vectors t1 and t2 which contains values from 0 to N-1.

    Example 1::
    -----------
        >>> t1 = torch.as_tensor([0, 1, 2, 4, 3, 6, 5, 7])
        >>> t2 = torch.as_tensor([0, 2, 1, 4, 3, 5, 6, 7])
        >>> indices = get_perm_indices(t1, t2)
        >>> (t1[indices] == t2).all().item()
        True
    """
    i1 = (t1[..., None, :] == t2[..., :, None]).int().argmax(dim=-2)
    return i1


def insert_at_indices(
    x: Tensor,
    indices: Union[Tensor, List, Number],
    values: Union[Number, Tensor],
) -> Tensor:
    """Insert value(s) in vector at specified indices.

    Example 1::
    -----------
        >>> x = torch.as_tensor([1, 1, 2, 2, 2, 3])
        >>> indices = torch.as_tensor([2, 5])
        >>> values = 4
        >>> insert_values(x, indices, values)
        tensor([1, 1, 4, 2, 2, 2, 4, 3])
    """
    if x.ndim != 1:
        raise ValueError(
            f"Invalid argument number of dims. (found {x.ndim=} but expected 1)"
        )

    device = x.device
    if isinstance(indices, (int, float, bool)):
        indices = torch.as_tensor([indices], device=device, dtype=torch.long)
    elif isinstance(indices, list):
        indices = torch.as_tensor(indices, device=device, dtype=torch.long)

    out = torch.empty((x.shape[0] + indices.shape[0]), dtype=x.dtype, device=device)
    indices = indices + torch.arange(
        indices.shape[0], device=indices.device, dtype=indices.dtype
    )
    out[indices] = values
    mask = torch.full((out.shape[0],), True, dtype=torch.bool)
    mask[indices] = False
    out[mask] = x
    return out


def remove_at_indices(
    x: Tensor,
    indices: Union[Tensor, List, Number],
) -> Tensor:
    """Remove value(s) in vector at specified indices."""
    if x.ndim != 1:
        raise ValueError(
            f"Invalid argument number of dims. (found {x.ndim=} but expected 1)"
        )

    device = x.device
    if isinstance(indices, (int, float, bool)):
        indices = torch.as_tensor([indices], device=device, dtype=torch.long)
    elif isinstance(indices, list):
        indices = torch.as_tensor(indices, device=device, dtype=torch.long)

    indices = indices + torch.arange(
        indices.shape[0], device=device, dtype=indices.dtype
    )
    mask = torch.full((x.shape[0],), True, dtype=torch.bool)
    mask[indices] = False
    out = x[mask]
    return out
