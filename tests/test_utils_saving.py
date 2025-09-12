#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from collections import Counter
from pathlib import Path
from unittest import TestCase

import torch

from torchoutil.utils.saving import to_builtin


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


if __name__ == "__main__":
    unittest.main()
