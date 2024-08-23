#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Generic, Optional, Sequence, Sized, TypeVar, Union

from torch.utils.data.dataset import Dataset

from pyoutil.typing.classes import SupportsGetItemLen

T = TypeVar("T", covariant=False)
U = TypeVar("U", covariant=False)


SizedDatasetLike = SupportsGetItemLen


class EmptyDataset(Dataset[None]):
    """Dataset placeholder. Raises StopIteration if __getitem__ is called."""

    def __getitem__(self, index) -> None:
        raise StopIteration

    def __len__(self) -> int:
        return 0


class TransformWrapper(Generic[T, U], Dataset[U]):
    def __init__(
        self,
        dataset: Sequence[T],
        transform: Optional[Callable[[T], U]],
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index) -> Union[T, U]:
        item = self._dataset[index]
        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise TypeError("Wrapped dataset is not Sized.")

    def unwrap(self) -> Sequence[T]:
        return self._dataset

    @property
    def transform(self) -> Optional[Callable[[T], U]]:
        return self._transform
