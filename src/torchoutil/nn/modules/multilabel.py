#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, List, Mapping, Optional, Union

import torch
from torch import Tensor, nn

from torchoutil.core.get import DeviceLike, DTypeLike
from torchoutil.nn.functional.multilabel import (
    T_Name,
    indices_to_multihot,
    indices_to_multinames,
    multihot_to_indices,
    multihot_to_multinames,
    multinames_to_indices,
    multinames_to_multihot,
    probs_to_indices,
    probs_to_multihot,
    probs_to_multinames,
)
from torchoutil.pyoutil.collections import dump_dict


class IndicesToMultihot(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.indices_to_multihot`.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        padding_idx: Optional[int] = None,
        device: DeviceLike = None,
        dtype: DTypeLike = torch.bool,
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


class IndicesToMultinames(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.indices_to_multinames`.
    """

    def __init__(
        self,
        idx_to_name: Mapping[int, T_Name],
        *,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name
        self.padding_idx = padding_idx

    def forward(
        self,
        indices: Union[List[List[int]], List[Tensor]],
    ) -> List[List[T_Name]]:
        names = indices_to_multinames(
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


class MultihotToMultinames(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.multihot_to_multinames`.
    """

    def __init__(
        self,
        idx_to_name: Mapping[int, T_Name],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        multihot: Tensor,
    ) -> List[List[T_Name]]:
        names = multihot_to_multinames(multihot, self.idx_to_name)
        return names


class MultinamesToIndices(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.multinames_to_indices`.
    """

    def __init__(
        self,
        idx_to_name: Mapping[int, T_Name],
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name

    def forward(
        self,
        names: List[List[T_Name]],
    ) -> List[List[int]]:
        indices = multinames_to_indices(names, self.idx_to_name)
        return indices


class MultinamesToMultihot(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.multinames_to_multihot`.
    """

    def __init__(
        self,
        idx_to_name: Mapping[int, T_Name],
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = torch.bool,
    ) -> None:
        super().__init__()
        self.idx_to_name = idx_to_name
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        names: List[List[T_Name]],
    ) -> Tensor:
        multihot = multinames_to_multihot(
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
        device: DeviceLike = None,
        dtype: DTypeLike = torch.bool,
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


class ProbsToMultinames(Generic[T_Name], nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.multilabel.probs_to_multinames`.
    """

    def __init__(
        self,
        threshold: Union[float, Tensor],
        idx_to_name: Mapping[int, T_Name],
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.idx_to_name = idx_to_name

    def forward(
        self,
        probs: Tensor,
    ) -> List[List[T_Name]]:
        names = probs_to_multinames(probs, self.threshold, self.idx_to_name)
        return names
