#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Generic, Iterator, Optional, TypeVar, Union

from torch.utils.data.dataset import Dataset, IterableDataset, Subset

from pyoutil.typing.classes import SupportsLenAndGetItem, SupportsLenAndGetItemAndIter

T = TypeVar("T", covariant=False)
U = TypeVar("U", covariant=False)

SizedDatasetLike = SupportsLenAndGetItem

T_SizedDatasetLike = TypeVar("T_SizedDatasetLike", bound=SupportsLenAndGetItem)
T_Dataset = TypeVar("T_Dataset", bound=Dataset)
T_SizedIterableDataset = TypeVar(
    "T_SizedIterableDataset", bound=SupportsLenAndGetItemAndIter
)


class EmptyDataset(Dataset[None]):
    """Dataset placeholder. Raises StopIteration if __getitem__ is called."""

    def __getitem__(self, index) -> None:
        raise StopIteration

    def __len__(self) -> int:
        return 0


class Wrapper(Generic[T_SizedDatasetLike, T], Dataset[T]):
    def __init__(
        self,
        dataset: T_SizedDatasetLike,
    ) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index) -> T:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def unwrap(self, recursive: bool = True) -> Union[SupportsLenAndGetItem, Dataset]:
        dataset = self.dataset
        continue_ = recursive and isinstance(dataset, Wrapper)
        while continue_:
            if not isinstance(dataset, (Wrapper, Subset)):
                break
            dataset = dataset.dataset
            continue_ = isinstance(dataset, Wrapper)
        return dataset


class IterableWrapper(
    Generic[T_SizedIterableDataset, T],
    IterableDataset[T],
    Wrapper[T_SizedIterableDataset, T],
):
    def __init__(
        self,
        dataset: T_SizedIterableDataset,
    ) -> None:
        IterableDataset.__init__(self)
        Wrapper.__init__(self, dataset)

    def __iter__(self) -> Iterator[T]:
        return iter(self.dataset)


class TransformWrapper(
    Generic[T_SizedDatasetLike, T, U],
    Wrapper[T_SizedDatasetLike, U],
):
    def __init__(
        self,
        dataset: T_SizedDatasetLike,
        transform: Optional[Callable[[T], U]],
        condition: Optional[Callable[[T, int], bool]] = None,
    ) -> None:
        super().__init__(dataset)
        self._transform = transform
        self._condition = condition

    def __getitem__(self, index) -> Union[T, U]:
        item = self.dataset[index]
        if (
            self._condition is None or self._condition(item, index)
        ) and self._transform is not None:
            item = self._transform(item)
        return item

    @property
    def transform(self) -> Optional[Callable[[T], U]]:
        return self._transform

    @property
    def condition(self) -> Optional[Callable[[T, int], bool]]:
        return self._condition
