#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    Sized,
    TypeVar,
    Union,
    runtime_checkable,
)

from torch.utils.data.dataset import Dataset

T = TypeVar("T", covariant=False)
U = TypeVar("U", covariant=False)


class EmptyDataset(Dataset[None]):
    """Dataset placeholder. Raises StopIteration if __getitem__ is called."""

    def __getitem__(self, index) -> None:
        raise StopIteration

    def __len__(self) -> int:
        return 0


@runtime_checkable
class SizedDatasetLike(Generic[T], Protocol):
    __getitem__: Callable[..., T]

    def __len__(self) -> int:
        ...


class TransformWrapper(Generic[T, U], Dataset[U]):
    def __init__(
        self,
        dataset: SizedDatasetLike[T],
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

    def unwrap(self) -> SizedDatasetLike[T]:
        return self._dataset

    @property
    def transform(self) -> Optional[Callable[[T], U]]:
        return self._transform
