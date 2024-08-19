#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Tensor, nn

from pyoutil.collections import dump_dict
from torchoutil.nn.functional.activation import log_softmax_multidim, softmax_multidim


class SoftmaxMultidim(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.activation.softmax_multidim`.
    """

    def __init__(
        self,
        dims: Union[Iterable[int], None] = (-1,),
    ) -> None:
        super().__init__()
        self.dims = dims

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        input = softmax_multidim(input, dims=self.dims)
        return input

    def extra_repr(self) -> str:
        return dump_dict(dict(dims=self.dims))


class LogSoftmaxMultidim(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.activation.softmax_multidim`.
    """

    def __init__(
        self,
        dims: Union[Iterable[int], None] = (-1,),
    ) -> None:
        super().__init__()
        self.dims = dims

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        input = log_softmax_multidim(input, dims=self.dims)
        return input

    def extra_repr(self) -> str:
        return dump_dict(dict(dims=self.dims))
