#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable

from torch import Tensor

from torchoutil.nn.functional.activation import softmax_multidim
from torchoutil.nn.functional.others import default_extra_repr
from torchoutil.nn.modules.typed import TModule


class SoftmaxMultidim(TModule[Tensor, Tensor]):
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
        return default_extra_repr(dims=self.dims)
