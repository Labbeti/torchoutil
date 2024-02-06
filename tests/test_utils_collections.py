#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.utils.collections import flat_dict_of_dict


class TestCollections(TestCase):
    def test_flat_dict_of_dict_example_1(self) -> None:
        x = {
            "a": 1,
            "b": {
                "a": 2,
                "b": 10,
            },
        }
        expected = {"a": 1, "b.a": 2, "b.b": 10}
        output = flat_dict_of_dict(x)
        self.assertDictEqual(output, expected)

    def test_flat_dict_of_dict_example_2(self) -> None:
        x = {"a": ["hello", "world"], "b": 3}
        expected = {"a.0": "hello", "a.1": "world", "b": 3}
        output = flat_dict_of_dict(x, flat_iterables=True)
        self.assertDictEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
