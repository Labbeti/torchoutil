#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Generic, Optional, TypeVar, Union

from torch.utils.data.dataset import Dataset, Subset

from pyoutil.typing.classes import SupportsLenAndGetItem

T = TypeVar("T", covariant=False)
U = TypeVar("U", covariant=False)

SizedDatasetLike = SupportsLenAndGetItem
TSizedDatasetLike = TypeVar("TSizedDatasetLike", bound=SizedDatasetLike)


class EmptyDataset(Dataset[None]):
    """Dataset placeholder. Raises StopIteration if __getitem__ is called."""

    def __getitem__(self, index) -> None:
        raise StopIteration

    def __len__(self) -> int:
        return 0


class Wrapper(Generic[TSizedDatasetLike, T], Dataset[T]):
    def __init__(
        self,
        dataset: TSizedDatasetLike,
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


class TransformWrapper(Generic[TSizedDatasetLike, T, U], Wrapper[TSizedDatasetLike, U]):
    def __init__(
        self,
        dataset: TSizedDatasetLike,
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
