#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Tensor, nn

from torchoutil.nn.functional.mask import masked_mean, masked_sum
from torchoutil.nn.functional.others import default_extra_repr


class MaskedMean(nn.Module):
    def __init__(self, dim: Union[None, int, Iterable[int]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: Tensor, non_pad_mask: Tensor) -> Tensor:
        reduced = masked_mean(tensor, non_pad_mask)
        return reduced

    def extra_repr(self) -> str:
        return default_extra_repr(
            dim=self.dim,
        )


class MaskedSum(nn.Module):
    def __init__(self, dim: Union[None, int, Iterable[int]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: Tensor, non_pad_mask: Tensor) -> Tensor:
        reduced = masked_sum(tensor, non_pad_mask)
        return reduced

    def extra_repr(self) -> str:
        return default_extra_repr(
            dim=self.dim,
        )
