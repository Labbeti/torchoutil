#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, List, Mapping, TypeVar, Union

import torch
from torch import Tensor, nn

from torchoutil.nn.functional.multilabel import (
    indices_to_multihot,
    indices_to_names,
    multihot_to_indices,
    multihot_to_names,
    names_to_indices,
    names_to_multihot,
    probs_to_indices,
    probs_to_multihot,
    probs_to_names,
)
from torchoutil.utils.collections import dump_dict

T = TypeVar("T")


class IndicesToMultihot(nn.Module):
    def __init__(
        self,
        num_classes: int,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = torch.bool,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        indices: Union[List[List[int]], List[Tensor]],
    ) -> Tensor:
        multihot = indices_to_multihot(
            indices, self.num_classes, self.device, self.dtype
        )
        return multihot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                num_classes=self.num_classes,
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_none=True,
        )


class IndicesToNames(nn.Module, Generic[T]):
    def __init__(
        self,
        idx_to_name: Mapping[int, T],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        indices: Union[List[List[int]], List[Tensor]],
    ) -> List[List[T]]:
        names = indices_to_names(indices, self.idx_to_name)
        return names


class MultihotToIndices(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        multihot: Tensor,
    ) -> List[List[int]]:
        names = multihot_to_indices(multihot)
        return names


class MultihotToNames(nn.Module, Generic[T]):
    def __init__(
        self,
        idx_to_name: Mapping[int, T],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        multihot: Tensor,
    ) -> List[List[T]]:
        names = multihot_to_names(multihot, self.idx_to_name)
        return names


class NamesToIndices(nn.Module, Generic[T]):
    def __init__(
        self,
        idx_to_name: Mapping[int, T],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        names: List[List[T]],
    ) -> List[List[int]]:
        indices = names_to_indices(names, self.idx_to_name)
        return indices


class NamesToMultihot(nn.Module, Generic[T]):
    def __init__(
        self,
        idx_to_name: Mapping[int, T],
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = torch.bool,
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        names: List[List[T]],
    ) -> Tensor:
        multihot = names_to_multihot(names, self.idx_to_name, self.device, self.dtype)
        return multihot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_none=True,
        )


class ProbsToIndices(nn.Module):
    def __init__(
        self,
        threshold: Union[float, Tensor],
    ) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        probs: Tensor,
    ) -> List[List[int]]:
        indices = probs_to_indices(probs, self.threshold)
        return indices


class ProbsToMultihot(nn.Module):
    def __init__(
        self,
        threshold: Union[float, Tensor],
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = torch.bool,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        probs: Tensor,
    ) -> Tensor:
        multihot = probs_to_multihot(probs, self.threshold, self.device, self.dtype)
        return multihot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_none=True,
        )


class ProbsToNames(nn.Module, Generic[T]):
    def __init__(
        self,
        threshold: Union[float, Tensor],
        idx_to_name: Mapping[int, T],
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.idx_to_name = idx_to_name

    def forward(
        self,
        probs: Tensor,
    ) -> List[List[T]]:
        names = probs_to_names(probs, self.threshold, self.idx_to_name)
        return names
