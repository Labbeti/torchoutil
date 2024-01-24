#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable

from torch import Tensor


def softmax_multidim(
    input: Tensor,
    dims: Iterable[int] = (-1,),
) -> Tensor:
    """A multi-dimensional version of torch.softmax."""
    dims = tuple(dims)
    x = input.exp()
    result = x / x.sum(dim=dims, keepdim=True)
    return result
