#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, Protocol, TypeVar, runtime_checkable

from torch.utils.data.dataset import Dataset

T = TypeVar("T", covariant=True)


class EmptyDataset(Dataset):
    """Dataset placeholder. Raises StopIteration if __getitem__ is called."""

    def __getitem__(self, idx) -> None:
        raise StopIteration

    def __len__(self) -> int:
        return 0


@runtime_checkable
class SizedDatasetLike(Generic[T], Protocol):
    def __getitem__(self, idx) -> T:
        ...

    def __len__(self) -> int:
        ...
