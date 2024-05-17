#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Tensor


def softmax_multidim(
    input: Tensor,
    dims: Union[Iterable[int], None] = (-1,),
) -> Tensor:
    """A multi-dimensional version of torch.softmax along multiple dimensions at the same time."""
    if dims is not None:
        dims = tuple(dims)

    x = input.exp()
    result = x / x.sum(dim=dims, keepdim=True)
    return result
