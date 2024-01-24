#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable

from torch import nn, Tensor

from extentorch.nn.functional.activation import softmax_multidim


class SoftmaxMultidim(nn.Module):
    def __init__(
        self,
        dims: Iterable[int] = (-1,),
    ) -> None:
        dims = tuple(dims)
        super().__init__()
        self.dims = dims

    def extra_repr(self) -> str:
        repr_params = {"dims": self.dims}
        repr_ = ", ".join(f"{name}={value}" for name, value in repr_params.items())
        return repr_

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        input = softmax_multidim(input, self.dims)
        return input
