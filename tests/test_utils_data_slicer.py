#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.utils.data.slicer import DatasetSlicerWrapper


class DummySequence:
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx):
        return 2 * idx


class TestSlicer(TestCase):
    def test_slicer(self) -> None:
        dset = DummySequence()
        slicer = DatasetSlicerWrapper(dset)

        indices = list(range(len(dset)))
        mask = [True] * len(dset)
        items = [dset[i] for i in indices]

        assert items == slicer[:]
        assert items == slicer[indices]
        assert items == slicer[mask]


if __name__ == "__main__":
    unittest.main()
