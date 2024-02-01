#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset


class EmptyDataset(Dataset):
    """Dataset placeholder. Raises StopIteration if __getitem__ is called."""

    def __getitem__(self, index) -> None:
        raise StopIteration

    def __len__(self) -> int:
        return 0
