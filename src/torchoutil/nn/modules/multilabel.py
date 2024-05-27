#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, List, Mapping, Optional, TypeVar, Union

import torch
from torch import Tensor, nn
from torch.types import Device

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
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.indices_to_multihot`.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        padding_idx: Optional[int] = None,
        device: Device = None,
        dtype: Union[torch.dtype, None] = torch.bool,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        indices: Union[List[List[int]], List[Tensor]],
    ) -> Tensor:
        multihot = indices_to_multihot(
            indices,
            self.num_classes,
            padding_idx=self.padding_idx,
            device=self.device,
            dtype=self.dtype,
        )
        return multihot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                num_classes=self.num_classes,
                padding_idx=self.padding_idx,
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_lst=(None,),
        )


class IndicesToNames(Generic[T], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.indices_to_names`.
    """

    def __init__(
        self,
        idx_to_name: Mapping[int, T],
        *,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name
        self.padding_idx = padding_idx

    def forward(
        self,
        indices: Union[List[List[int]], List[Tensor]],
    ) -> List[List[T]]:
        names = indices_to_names(
            indices,
            self.idx_to_name,
            padding_idx=self.padding_idx,
        )
        return names

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                padding_idx=self.padding_idx,
            ),
            ignore_lst=(None,),
        )


class MultihotToIndices(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.multihot_to_indices`.
    """

    def __init__(self, *, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.padding_idx = padding_idx

    def forward(
        self,
        multihot: Tensor,
    ) -> List[List[int]]:
        names = multihot_to_indices(multihot, padding_idx=self.padding_idx)
        return names

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                padding_idx=self.padding_idx,
            ),
            ignore_lst=(None,),
        )


class MultihotToNames(Generic[T], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.multihot_to_names`.
    """

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


class NamesToIndices(Generic[T], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.names_to_indices`.
    """

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


class NamesToMultihot(Generic[T], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.names_to_multihot`.
    """

    def __init__(
        self,
        idx_to_name: Mapping[int, T],
        *,
        device: Device = None,
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
        multihot = names_to_multihot(
            names,
            self.idx_to_name,
            device=self.device,
            dtype=self.dtype,
        )
        return multihot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_lst=(None,),
        )


class ProbsToIndices(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.probs_to_indices`.
    """

    def __init__(
        self,
        threshold: Union[float, Tensor],
        *,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.padding_idx = padding_idx

    def forward(
        self,
        probs: Tensor,
    ) -> List[List[int]]:
        indices = probs_to_indices(probs, self.threshold, padding_idx=self.padding_idx)
        return indices


class ProbsToMultihot(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.probs_to_multihot`.
    """

    def __init__(
        self,
        threshold: Union[float, Tensor],
        *,
        device: Device = None,
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
        multihot = probs_to_multihot(
            probs,
            self.threshold,
            device=self.device,
            dtype=self.dtype,
        )
        return multihot

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_lst=(None,),
        )


class ProbsToNames(Generic[T], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.probs_to_names`.
    """

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
