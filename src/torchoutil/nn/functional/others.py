#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Generator, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.types import Number

from torchoutil.nn.functional.pad import pad_dim


def find(
    x: Tensor,
    value: Any,
    default: Union[None, Tensor, int, float] = None,
    dim: int = -1,
) -> Tensor:
    """Return the index of the first occurrence of value in a tensor."""
    if x.ndim == 0:
        raise ValueError(
            f"Function find does not supports 0-d tensors. (found {x.ndim=} == 0)"
        )
    mask = x.eq(value)
    contains_eos = mask.any(dim=dim)
    indices_eos = mask.int().argmax(dim=dim)

    if default is None:
        if not contains_eos.all():
            raise RuntimeError(f"Cannot find {value=} in tensor.")
        return indices_eos
    else:
        output = torch.where(contains_eos, indices_eos, default)
        return output


def move_to_rec(
    x: Any,
    *args,
    predicate: Optional[Callable[[Any], bool]] = None,
    **kwargs,
) -> Any:
    """Move all modules and tensors recursively to dtype or device."""
    if isinstance(x, (Tensor, nn.Module)):
        if predicate is None or predicate(x):
            return x.to(*args, **kwargs)
        else:
            return x
    elif isinstance(x, (str, float, int, bool, complex)):
        return x
    elif isinstance(x, Mapping):
        return {
            k: move_to_rec(v, predicate=predicate, *args, **kwargs)
            for k, v in x.items()
        }
    elif isinstance(x, Iterable):
        generator = (move_to_rec(xi, predicate=predicate, *args, **kwargs) for xi in x)
        if isinstance(x, Generator):
            return generator
        elif isinstance(x, tuple):
            return tuple(generator)
        elif isinstance(x, list):
            return list(generator)
        else:
            return list(generator)
    else:
        return x


def cat_padded_batch(
    x1: Tensor,
    x1_lens: Tensor,
    x2: Tensor,
    x2_lens: Tensor,
    seq_dim: int,
    batch_dim: int = 0,
) -> Tuple[Tensor, Tensor]:
    assert x1.ndim == x2.ndim
    assert x1_lens.ndim == x2_lens.ndim == 1
    assert (
        x1.shape[batch_dim]
        == x2.shape[batch_dim]
        == x1_lens.shape[0]
        == x2_lens.shape[0]
    )

    x12_lens = x1_lens + x2_lens
    sum_size_12 = x1.shape[seq_dim] + x2.shape[seq_dim]

    x12 = pad_dim(x1, sum_size_12, dim=seq_dim)
    kwd: dict[str, Any] = dict(device=x1.device, dtype=torch.long)
    indices = torch.arange(x2_lens.max().item(), **kwd)

    unsq_x1_lens = x1_lens
    ndim = x1.ndim
    for i in range(ndim):
        if i != (seq_dim % ndim):
            indices = indices.unsqueeze(dim=i)
        if i != (batch_dim % ndim):
            unsq_x1_lens = unsq_x1_lens.unsqueeze(dim=i)

    expand_size = list(x2.shape)
    expand_size[seq_dim] = -1
    indices = indices.expand(*expand_size)
    indices = indices + unsq_x1_lens
    x12.scatter_(seq_dim, indices, x2)

    max_size_12 = int(x12_lens.max().item())
    if max_size_12 < sum_size_12:
        slices = [slice(None) for _ in range(ndim)]
        slices[seq_dim] = slice(max_size_12)
        x12 = x12[slices]

    return x12, x12_lens


def is_scalar(x: Union[Number, Tensor]) -> bool:
    return isinstance(x, Number) or (isinstance(x, Tensor) and x.ndim == 0)


def default_extra_repr(
    join: str = ", ",
    fmt: str = "{name}={value}",
    ignore_if_none: bool = False,
    **kwargs,
) -> str:
    result = join.join(
        fmt.format(name=name, value=value)
        for name, value in kwargs.items()
        if not (value is None and ignore_if_none)
    )
    return result
