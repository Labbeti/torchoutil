#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Tensor, nn

from torchoutil.nn.functional.activation import log_softmax_multidim, softmax_multidim
from torchoutil.pyoutil.collections import dump_dict


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
        return softmax_multidim(input, dims=self.dims)

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
        return log_softmax_multidim(input, dims=self.dims)

    def extra_repr(self) -> str:
        return dump_dict(dict(dims=self.dims))
