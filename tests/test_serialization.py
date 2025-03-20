#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from collections import Counter
from pathlib import Path
from typing import List, Tuple
from unittest import TestCase

import torch

from torchoutil.serialization.common import (
    SavingBackend,
    _fpath_to_saving_backend,
    to_builtin,
)


class TestSaving(TestCase):
    def test_examples(self) -> None:
        x = [
            [
                torch.arange(3)[None],
                "a",
                Path("./path"),
                Counter(["a", "b", "a", "c", "a"]),
                (),
            ],
        ]
        expected = [[[list(range(3))], "a", "path", {"a": 3, "b": 1, "c": 1}, []]]
        result = to_builtin(x)
        assert result == expected, f"{result=}; {expected=}"

    def test_backend(self) -> None:
        tests: List[Tuple[str, SavingBackend]] = [
            ("test.json", "json"),
            ("test.json.yaml", "yaml"),
            ("test.yaml.json", "json"),
        ]

        for fpath, expected_backend in tests:
            backend = _fpath_to_saving_backend(fpath)
            assert backend == expected_backend


if __name__ == "__main__":
    unittest.main()
