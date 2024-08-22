#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Generic, Iterable, List, Sequence, TypeVar, Union, overload

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

from pyoutil.typing import is_iterable_bool, is_iterable_int
from torchoutil.types import is_bool_tensor1d, is_integer_tensor1d
from torchoutil.types._hints import BoolTensor1D, Tensor1D

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)

Indices = Union[Iterable[bool], Iterable[int], None, slice, Tensor1D]


class DatasetSlicer(Dataset[T], Generic[T, U]):
    def __init__(
        self,
        *,
        add_slice_support: bool = True,
        add_indices_support: bool = True,
        add_mask_support: bool = True,
    ) -> None:
        super().__init__()
        self._add_slice_support = add_slice_support
        self._add_indices_support = add_indices_support
        self._add_mask_support = add_mask_support

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_item(self, index: U, *args) -> T:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: U) -> T:
        ...

    @overload
    def __getitem__(self, index: Indices) -> List[T]:
        ...

    def __getitem__(self, index) -> Union[T, List[T]]:
        if isinstance(index, tuple) and len(index) > 1:
            index, *args = index
        else:
            args = ()

        if isinstance(index, int) or (isinstance(index, Tensor) and index.ndim == 0):
            return self.get_item(index, *args)

        elif isinstance(index, slice):
            return self.get_items_slice(index, *args)

        elif is_iterable_bool(index) or is_bool_tensor1d(index):
            return self.get_items_mask(index, *args)

        elif is_iterable_int(index) or is_integer_tensor1d(index):
            return self.get_items_indices(index, *args)

        else:
            raise TypeError(f"Invalid argument type {type(index)=} with {args=}.")

    def __getitems__(
        self,
        indices: Indices,
        *args,
    ) -> List[T]:
        return self.__getitem__(indices, *args)

    def get_items_indices(
        self,
        indices: Union[Iterable[int], Tensor1D],
        *args,
    ) -> List[T]:
        if self._add_indices_support:
            return [self.get_item(idx, *args) for idx in indices]
        else:
            return self.get_item(indices, *args)

    def get_items_mask(
        self,
        mask: Union[Iterable[bool], BoolTensor1D],
        *args,
    ) -> List[T]:
        if self._add_mask_support:
            mask = torch.as_tensor(mask, dtype=torch.bool)
            if len(mask) > 0 and len(mask) != len(self):
                raise ValueError(
                    f"Invalid mask size {len(mask)}. (expected {len(self)})"
                )
            indices = _where_1d(mask)
            return self.get_items_indices(indices, *args)
        else:
            return self.get_item(mask, *args)

    def get_items_slice(
        self,
        slice_: slice,
        *args,
    ) -> List[T]:
        if self._add_slice_support:
            return self.get_items_indices(range(len(self))[slice_], *args)
        else:
            return self.get_item(slice_, *args)


class DatasetSlicerWrapper(DatasetSlicer):
    def __init__(
        self,
        dataset: Sequence[T],
        *,
        add_slice_support: bool = True,
        add_indices_support: bool = True,
        add_mask_support: bool = True,
    ) -> None:
        """Wrap a sequence to support slice, indices and mask arguments types."""
        super().__init__(
            add_slice_support=add_slice_support,
            add_indices_support=add_indices_support,
            add_mask_support=add_mask_support,
        )
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def get_item(self, index, *args) -> T:
        return self.dataset.__getitem__(index, *args)


def _where_1d(mask: Union[Iterable[bool], BoolTensor1D]) -> Tensor1D:
    if is_iterable_bool(mask):
        mask = torch.as_tensor(mask)
    return torch.where(mask)[0]
