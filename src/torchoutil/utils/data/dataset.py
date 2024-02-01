#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, Protocol, TypeVar

from torch.utils.data.dataset import Dataset

T = TypeVar("T", covariant=True)


class EmptyDataset(Dataset):
    """Dataset placeholder. Raises StopIteration if __getitem__ is called."""

    def __getitem__(self, index) -> None:
        raise StopIteration

    def __len__(self) -> int:
        return 0


class SizedDatasetLike(Protocol, Generic[T]):
    def __getitem__(self, index) -> T:
        ...

    def __len__(self) -> int:
        ...
