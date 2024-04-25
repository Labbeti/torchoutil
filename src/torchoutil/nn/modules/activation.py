#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Tensor, nn

from torchoutil.nn.functional.activation import softmax_multidim
from torchoutil.utils.collections import dump_dict


class SoftmaxMultidim(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.activation.softmax_multidim`.
    """

    def __init__(
        self,
        dims: Union[Iterable[int], None] = (-1,),
    ) -> None:
        if dims is not None:
            dims = tuple(dims)

        super().__init__()
        self.dims = dims

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        input = softmax_multidim(input, self.dims)
        return input

    def extra_repr(self) -> str:
        return dump_dict(dict(dims=self.dims))
