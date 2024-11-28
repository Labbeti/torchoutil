#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from typing import Union
from unittest import TestCase

from torchoutil.utils.data.slicer import DatasetSlicer, DatasetSlicerWrapper


class DummySequence1:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int):
        return 2 * idx


class DummySequence2:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: Union[tuple, int]):
        if isinstance(idx, tuple):
            idx, *args = idx
        else:
            args = ()
        return (2 * idx,) + tuple(args)


class DummySequence3(DatasetSlicer):
    def __len__(self) -> int:
        return 5

    def get_item(self, idx: int) -> int:
        return idx * 3


class TestSlicer(TestCase):
    def test_slicer_wrapper(self) -> None:
        ds = DummySequence1()
        slicer = DatasetSlicerWrapper(ds)

        indices = list(range(len(ds)))
        mask = [True] * len(ds)
        items = [ds[i] for i in indices]

        assert items == slicer[:]
        assert items == slicer[indices]
        assert items == slicer[mask]

    def test_slicer_wrapper_args(self) -> None:
        ds = DummySequence2()
        slicer = DatasetSlicerWrapper(ds)

        indices = list(range(len(ds)))
        mask = [True] * len(ds)
        items = [ds[i] for i in indices]

        assert items == slicer[:]
        assert items == slicer[indices]
        assert items == slicer[mask]

        assert [(2, "a", "b"), (4, "a", "b")] == slicer[1:3, "a", "b"]

    def test_slicer_base_class(self) -> None:
        ds = DummySequence3()

        indices = list(range(len(ds)))
        random.shuffle(indices)

        mask = [random.random() > 0.5 for _ in range(len(ds))]

        assert ds[:] == [ds[i] for i in range(len(ds))]
        assert ds[indices] == [ds[idx] for idx in indices]
        assert ds[mask] == [ds[i] for i in range(len(ds)) if mask[i]]


if __name__ == "__main__":
    unittest.main()
