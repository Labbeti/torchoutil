#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable

from torch import Tensor, nn

from torchoutil.nn.functional.activation import softmax_multidim
from torchoutil.nn.functional.others import dump_dict


class SoftmaxMultidim(nn.Module):
    def __init__(
        self,
        dims: Iterable[int] = (-1,),
    ) -> None:
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
        return dump_dict(dims=self.dims)
