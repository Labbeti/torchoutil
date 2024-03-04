#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, List, Mapping, TypeVar, Union

import torch
from torch import Tensor, nn

from torchoutil.nn.functional.multiclass import (
    indices_to_names,
    indices_to_onehot,
    names_to_indices,
    names_to_onehot,
    onehot_to_indices,
    onehot_to_names,
    probs_to_indices,
    probs_to_names,
    probs_to_onehot,
)
from torchoutil.utils.collections import dump_dict

T = TypeVar("T")


class IndicesToOneHot(nn.Module):
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
        indices: Union[List[int], Tensor],
    ) -> Tensor:
        onehot = indices_to_onehot(indices, self.num_classes, self.device, self.dtype)
        return onehot

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
        indices: Union[List[int], Tensor],
    ) -> List[T]:
        names = indices_to_names(indices, self.idx_to_name)
        return names


class OneHotToIndices(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        onehot: Tensor,
    ) -> List[int]:
        names = onehot_to_indices(onehot)
        return names


class OneHotToNames(nn.Module, Generic[T]):
    def __init__(
        self,
        idx_to_name: Mapping[int, T],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        onehot: Tensor,
    ) -> List[T]:
        names = onehot_to_names(onehot, self.idx_to_name)
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
        names: List[T],
    ) -> List[int]:
        indices = names_to_indices(names, self.idx_to_name)
        return indices


class NamesToOneHot(nn.Module, Generic[T]):
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
        names: List[T],
    ) -> Tensor:
        onehot = names_to_onehot(names, self.idx_to_name, self.device, self.dtype)
        return onehot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_none=True,
        )


class ProbsToIndices(nn.Module):
    def forward(
        self,
        probs: Tensor,
    ) -> List[int]:
        indices = probs_to_indices(probs)
        return indices


class ProbsToOneHot(nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = torch.bool,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        probs: Tensor,
    ) -> Tensor:
        onehot = probs_to_onehot(probs, self.device, self.dtype)
        return onehot

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
        idx_to_name: Mapping[int, T],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        probs: Tensor,
    ) -> List[T]:
        names = probs_to_names(probs, self.idx_to_name)
        return names
