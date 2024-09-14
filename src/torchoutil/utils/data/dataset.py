#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar, Union

from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.dataset import Subset as TorchSubset

from torchoutil import LongTensor1D
from torchoutil.pyoutil.collections import is_sorted
from torchoutil.pyoutil.typing.classes import (
    SupportsLenAndGetItem,
    SupportsLenAndGetItemAndIter,
)

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)

SizedDatasetLike = SupportsLenAndGetItem
SizedIterableDatasetLike = SupportsLenAndGetItemAndIter

T_Dataset = TypeVar("T_Dataset", bound=Dataset)
T_SizedDatasetLike = TypeVar("T_SizedDatasetLike", bound=SupportsLenAndGetItem)
T_SizedIterableDataset = TypeVar(
    "T_SizedIterableDataset",
    bound=SupportsLenAndGetItemAndIter,
)


class EmptyDataset(Dataset[None]):
    """Dataset placeholder. Raises StopIteration if __getitem__ is called."""

    def __getitem__(self, idx, /) -> None:
        raise StopIteration

    def __len__(self) -> int:
        return 0


class Wrapper(Generic[T], Dataset[T]):
    def __init__(self, dataset: SupportsLenAndGetItem[T]) -> None:
        Dataset.__init__(self)
        self.dataset = dataset

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx, /) -> T:
        raise NotImplementedError

    def unwrap(self, recursive: bool = True) -> Union[SupportsLenAndGetItem, Dataset]:
        dataset = self.dataset
        continue_ = recursive and isinstance(dataset, Wrapper)
        while continue_:
            if not isinstance(dataset, (Wrapper, TorchSubset)):
                break
            dataset = dataset.dataset
            continue_ = isinstance(dataset, Wrapper)
        return dataset

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.dataset)})"


class IterableWrapper(Generic[T], IterableDataset[T], Wrapper[T]):
    def __init__(self, dataset: SupportsLenAndGetItem[T]) -> None:
        IterableDataset.__init__(self)
        Wrapper.__init__(self, dataset)

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        ...

    def _get_dataset_iter(self) -> Iterator[T]:
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

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Union[T, U]:
        assert isinstance(idx, int)
        item = self.dataset[idx]
        if self._transform is not None and (
            self._condition is None or self._condition(item, idx)
        ):
            item = self._transform(item)
        return item

    @property
    def transform(self) -> Optional[Callable[[T], U]]:
        return self._transform

    @property
    def condition(self) -> Optional[Callable[[T, int], bool]]:
        return self._condition


class IterableTransformWrapper(IterableWrapper[T], Generic[T, U]):
    def __init__(
        self,
        dataset: SupportsLenAndGetItem[T],
        transform: Optional[Callable[[T], U]],
        condition: Optional[Callable[[T, int], bool]] = None,
    ) -> None:
        super().__init__(dataset)
        self._transform = transform
        self._condition = condition

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[Union[T, U]]:
        it = super()._get_dataset_iter()
        for i, item in enumerate(it):
            if self._transform is not None and (
                self._condition is None or self._condition(item, i)
            ):
                item = self._transform(item)
            yield item
        return

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
        indices: Union[Iterable[int], LongTensor1D],
    ) -> None:
        if isinstance(indices, LongTensor1D):
            indices = indices.tolist()
        else:
            indices = list(indices)

        if not all(idx >= 0 for idx in indices) or not is_sorted(indices):
            msg = f"Invalid argument {indices=}. (expected a sorted list of positive integers)"
            raise ValueError(msg)

        super().__init__(dataset)
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[T]:
        it = super()._get_dataset_iter()

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


class Subset(Generic[T], TorchSubset[T], Wrapper[T]):
    def __init__(self, dataset: SizedDatasetLike[T], indices: Iterable[int]) -> None:
        indices = list(indices)
        TorchSubset.__init__(self, dataset, indices)  # type: ignore
        Wrapper.__init__(self, dataset)
