#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

from torchoutil.utils.collections import (
    flat_dict_of_dict,
    flat_list_of_list,
    intersect_lists,
    list_dict_to_dict_list,
    unflat_dict_of_dict,
    unflat_list_of_list,
)
from torchoutil.utils.type_checks import is_list_list_str, is_list_str


class TestFlatDict(TestCase):
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

    def test_unflat_dict_example_1(self) -> None:
        x = {
            "a.a": 1,
            "b.a": 2,
            "b.b": 3,
            "c": 4,
        }
        y = {"a": {"a": 1}, "b": {"a": 2, "b": 3}, "c": 4}

        y_hat = unflat_dict_of_dict(x)
        self.assertEqual(y_hat, y)


class TestListDictToDictList(TestCase):
    def test_example_1(self) -> None:
        lst = [{"a": 1, "b": 2}, {"a": 4, "b": 3, "c": 5}]
        output = list_dict_to_dict_list(lst, default_val=0)
        self.assertDictEqual(output, {"a": [1, 4], "b": [2, 3], "c": [0, 5]})

    def test_example_2(self) -> None:
        lst = [{"a": 1, "b": 2}, {"a": 4, "b": 3, "c": 5}]

        with self.assertRaises(ValueError):
            list_dict_to_dict_list(
                lst,
                default_val=None,
                key_mode="same",
            )

    def test_example_3(self) -> None:
        lst = [{"a": 1, "b": 2, "c": 3}, {"a": 11, "b": 22, "c": 33}]
        output = list_dict_to_dict_list(
            lst,
            default_val=None,
            key_mode="same",
        )
        self.assertDictEqual(output, {"a": [1, 11], "b": [2, 22], "c": [3, 33]})


class TestIntersectLists(TestCase):
    def test_example_1(self) -> None:
        input_ = [["a", "b", "b", "c"], ["c", "d", "b", "a"], ["b", "a", "a", "e"]]
        expected = ["a", "b"]

        output = intersect_lists(input_)
        self.assertListEqual(output, expected)


class TestFlatList(TestCase):
    def test_example_1(self) -> None:
        lst = [
            list(map(str, range(random.randint(0, 100))))
            for _ in range(random.randint(0, 10))
        ]
        for sublst in lst:
            random.shuffle(sublst)
        random.shuffle(lst)

        self.assertTrue(is_list_list_str(lst))

        flatten, sizes = flat_list_of_list(lst)

        self.assertTrue(is_list_str(flatten))
        self.assertEqual(len(lst), len(sizes))
        self.assertEqual(len(flatten), sum(sizes))

        unflat = unflat_list_of_list(flatten, sizes)

        self.assertTrue(is_list_list_str(unflat))
        self.assertEqual(len(lst), len(unflat))
        self.assertListEqual(lst, unflat)


if __name__ == "__main__":
    unittest.main()
