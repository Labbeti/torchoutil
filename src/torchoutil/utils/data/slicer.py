#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, List, Tuple, TypeVar, Union, final, overload

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

from torchoutil.pyoutil.typing import is_iterable_bool, is_iterable_int
from torchoutil.pyoutil.typing.classes import SupportsLenAndGetItem
from torchoutil.types import is_bool_tensor1d, is_integer_tensor1d
from torchoutil.types._hints import BoolTensor, BoolTensor1D, Tensor1D
from torchoutil.utils.data.dataset import Wrapper

T = TypeVar("T", covariant=False)
U = TypeVar("U", covariant=False)

Indices = Union[Iterable[bool], Iterable[int], None, slice, Tensor1D]


class DatasetSlicer(Generic[T], ABC, Dataset[T]):
    def __init__(
        self,
        *,
        add_slice_support: bool = True,
        add_indices_support: bool = True,
        add_mask_support: bool = True,
    ) -> None:
        Dataset.__init__(self)
        self._add_slice_support = add_slice_support
        self._add_indices_support = add_indices_support
        self._add_mask_support = add_mask_support

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_item(self, idx, *args):
        raise NotImplementedError

    @overload
    def __getitem__(self, idx: int, /) -> T:
        ...

    @overload
    def __getitem__(self, idx: Indices, /) -> List[T]:
        ...

    @overload
    def __getitem__(self, idx: Tuple[Any, ...], /) -> Any:
        ...

    @final
    def __getitem__(self, idx, /):
        if isinstance(idx, tuple) and len(idx) > 1:
            idx, *args = idx
        else:
            args = ()

        if isinstance(idx, int) or (isinstance(idx, Tensor) and idx.ndim == 0):
            return self.get_item(idx, *args)

        elif isinstance(idx, slice):
            return self.get_items_slice(idx, *args)

        elif is_iterable_bool(idx) or is_bool_tensor1d(idx):
            return self.get_items_mask(idx, *args)

        elif is_iterable_int(idx) or is_integer_tensor1d(idx):
            return self.get_items_indices(idx, *args)

        else:
            raise TypeError(f"Invalid argument type {type(idx)=} with {args=}.")

    @final
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
            if not isinstance(mask, Tensor):
                mask = torch.as_tensor(list(mask), dtype=torch.bool)  # type: ignore
            if len(mask) > 0 and len(mask) != len(self):  # type: ignore
                raise ValueError(
                    f"Invalid mask size {len(mask)}. (expected {len(self)})"  # type: ignore
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


class DatasetSlicerWrapper(Generic[T], DatasetSlicer[T], Wrapper[T]):
    def __init__(
        self,
        dataset: SupportsLenAndGetItem[T],
        *,
        add_slice_support: bool = True,
        add_indices_support: bool = True,
        add_mask_support: bool = True,
    ) -> None:
        """Wrap a sequence to support slice, indices and mask arguments types."""
        DatasetSlicer.__init__(
            self,
            add_slice_support=add_slice_support,
            add_indices_support=add_indices_support,
            add_mask_support=add_mask_support,
        )
        Wrapper.__init__(self, dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_item(self, idx: int, *args) -> T:
        # note: we need to split calls here, because self.dataset[idx] give an int as argument while self.dataset[idx, *args] always gives a tuple even if args == ()
        if len(args) == 0:
            return self.dataset[idx]
        else:
            return self.dataset[idx, *args]


def _where_1d(mask: Union[Iterable[bool], BoolTensor1D]) -> Tensor1D:
    if not isinstance(mask, BoolTensor):
        mask = torch.as_tensor(list(mask), dtype=torch.bool)  # type: ignore
    return torch.where(mask)[0]  # type: ignore
