#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

from torch import Tensor, nn

from torchoutil.nn.functional.mask import masked_mean, masked_sum
from torchoutil.utils.collections import dump_dict


class MaskedMean(nn.Module):
    def __init__(self, dim: Union[None, int, Iterable[int]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: Tensor, non_pad_mask: Tensor) -> Tensor:
        reduced = masked_mean(tensor, non_pad_mask, self.dim)
        return reduced

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
            ),
            ignore_none=True,
        )


class MaskedSum(nn.Module):
    def __init__(self, dim: Union[None, int, Iterable[int]] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: Tensor, non_pad_mask: Tensor) -> Tensor:
        reduced = masked_sum(tensor, non_pad_mask, self.dim)
        return reduced

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
            ),
            ignore_none=True,
        )
