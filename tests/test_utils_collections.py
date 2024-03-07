#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.utils.collections import flat_dict_of_dict, list_dict_to_dict_list


class TestCollections(TestCase):
    def test_example_1(self) -> None:
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

    def test_example_2(self) -> None:
        x = {"a": ["hello", "world"], "b": 3}
        expected = {"a.0": "hello", "a.1": "world", "b": 3}

        output = flat_dict_of_dict(x, flat_iterables=True)
        self.assertDictEqual(output, expected)

    def test_example_3(self) -> None:
        x = {"a": {"b": 1}, "a.b": 2}
        expected = {"a.b": 2}

        output = flat_dict_of_dict(x)
        assert output == expected

    def test_example_4(self) -> None:
        x = {"a": {"b": 1}, "a.b": 2}
        self.assertRaises(ValueError, flat_dict_of_dict, x, overwrite=False)


class TestListDictToDictList(TestCase):
    def test_example_1(self) -> None:
        lst = [{"a": 1, "b": 2}, {"a": 4, "b": 3, "c": 5}]
        output = list_dict_to_dict_list(lst, default_val=0)
        self.assertDictEqual(output, {"a": [1, 4], "b": [2, 3], "c": [0, 5]})


if __name__ == "__main__":
    unittest.main()
