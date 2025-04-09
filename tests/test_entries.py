#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import unittest
from unittest import TestCase

from torchoutil.entries import print_safe_rmdir, print_tree
from torchoutil.hub.paths import get_tmp_dir


class TestEntries(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmpdir = get_tmp_dir().joinpath("torchoutil_tests")
        tmpdir.mkdir(parents=True, exist_ok=True)
        cls.tmpdir = tmpdir

        root = tmpdir.joinpath("test_print_safe_rmdir")
        root.joinpath("a", "a1").mkdir(parents=True)
        root.joinpath("a", "a2").mkdir()
        root.joinpath("b").mkdir()
        root.joinpath("c").mkdir()

        cls.safe_rmdir_root = root

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.safe_rmdir_root.is_dir():
            shutil.rmtree(cls.safe_rmdir_root)

    def test_print_safe_rmdir(self) -> None:
        print_safe_rmdir(self.safe_rmdir_root)

    def test_print_tree(self) -> None:
        print_tree(".", max_depth=0)


if __name__ == "__main__":
    unittest.main()
