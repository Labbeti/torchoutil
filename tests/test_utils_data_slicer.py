#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from typing import Union
from unittest import TestCase

from torchoutil.utils.data.slicer import DatasetSlicerWrapper


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


class TestSlicer(TestCase):
    def test_slicer_1(self) -> None:
        dset = DummySequence1()
        slicer = DatasetSlicerWrapper(dset)

        indices = list(range(len(dset)))
        mask = [True] * len(dset)
        items = [dset[i] for i in indices]

        assert items == slicer[:]
        assert items == slicer[indices]
        assert items == slicer[mask]

    def test_slicer_2(self) -> None:
        dset = DummySequence2()
        slicer = DatasetSlicerWrapper(dset)

        indices = list(range(len(dset)))
        mask = [True] * len(dset)
        items = [dset[i] for i in indices]

        assert items == slicer[:]
        assert items == slicer[indices]
        assert items == slicer[mask]

        assert [(2, "a", "b"), (4, "a", "b")] == slicer[1:3, "a", "b"]


if __name__ == "__main__":
    unittest.main()
