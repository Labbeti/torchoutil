#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar, Union

from torch.utils.data.dataset import Dataset, IterableDataset, Subset

from pyoutil.collections import is_sorted
from pyoutil.typing.classes import SupportsLenAndGetItem, SupportsLenAndGetItemAndIter
from torchoutil import LongTensor1D

T = TypeVar("T", covariant=False)
U = TypeVar("U", covariant=False)

SizedDatasetLike = SupportsLenAndGetItem
SizedIterableDatasetLike = SupportsLenAndGetItemAndIter

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


class Wrapper(Generic[T], Dataset[T]):
    def __init__(
        self,
        dataset: SupportsLenAndGetItem[T],
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


class IterableWrapper(Generic[T], Wrapper[T], IterableDataset[T]):
    def __init__(self, dataset: SupportsLenAndGetItem[T]) -> None:
        Wrapper.__init__(self, dataset)
        IterableDataset.__init__(self)

    def __iter__(self) -> Iterator[T]:
        if hasattr(self.dataset, "__iter__"):
            it = iter(self.dataset)
        else:
            it = (self.dataset[i] for i in range(len(self.dataset)))
        return it


class TransformWrapper(Generic[T, U], Wrapper[T]):
    def __init__(
        self,
        dataset: SupportsLenAndGetItem[T],
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


class IterableSubset(IterableWrapper[T], Generic[T]):
    def __init__(
        self,
        dataset: SupportsLenAndGetItem[T],
        indices: Iterable[int] | LongTensor1D,
    ) -> None:
        if isinstance(indices, LongTensor1D):
            indices = indices.tolist()
        else:
            indices = list(indices)

        assert all(idx >= 0 for idx in indices)
        assert is_sorted(indices)

        super().__init__(dataset)
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[T]:
        it = super().__iter__()

        cur_idx = 0
        item = next(it)

        for idx in self._indices:
            if cur_idx == idx:
                yield item
                continue

            while cur_idx < idx:
                cur_idx += 1
                item = next(it)

            yield item
        return
