#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

from pyoutil.collections import (
    flat_dict_of_dict,
    flat_list_of_list,
    flatten,
    intersect_lists,
    list_dict_to_dict_list,
    unflat_dict_of_dict,
    unflat_list_of_list,
)
from pyoutil.re import get_key_fn
from pyoutil.typing import is_list_list_str, is_list_str


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
        assert output == expected

    def test_example_2(self) -> None:
        x = {"a": ["hello", "world"], "b": 3}
        expected = {"a.0": "hello", "a.1": "world", "b": 3}

        output = flat_dict_of_dict(x, flat_iterables=True)
        assert output == expected

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
        assert y_hat == y


class TestListDictToDictList(TestCase):
    def test_example_1(self) -> None:
        lst = [{"a": 1, "b": 2}, {"a": 4, "b": 3, "c": 5}]
        output = list_dict_to_dict_list(lst, default_val=0)
        expected = {"a": [1, 4], "b": [2, 3], "c": [0, 5]}
        assert output == expected

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
        expected = {"a": [1, 11], "b": [2, 22], "c": [3, 33]}
        assert output == expected


class TestIntersectLists(TestCase):
    def test_example_1(self) -> None:
        input_ = [["a", "b", "b", "c"], ["c", "d", "b", "a"], ["b", "a", "a", "e"]]
        expected = ["a", "b"]
        output = intersect_lists(input_)
        assert output == expected


class TestFlatList(TestCase):
    def test_example_1(self) -> None:
        lst = [
            list(map(str, range(random.randint(0, 100))))
            for _ in range(random.randint(0, 10))
        ]
        for sublst in lst:
            random.shuffle(sublst)
        random.shuffle(lst)

        assert is_list_list_str(lst)

        flatten, sizes = flat_list_of_list(lst)
        assert is_list_str(flatten)
        assert len(lst) == len(sizes)
        assert len(flatten) == sum(sizes)

        unflat = unflat_list_of_list(flatten, sizes)
        assert is_list_list_str(unflat)
        assert len(lst) == len(unflat)
        assert lst == unflat


class TestGetKeyFn(TestCase):
    def test_example_1(self) -> None:
        lst = ["a", "abc", "aa", "abcd"]
        patterns = ["^ab.*"]  # sort list with elements starting with 'ab' first
        result = list(sorted(lst, key=get_key_fn(patterns)))
        expected = ["abc", "abcd", "a", "aa"]
        assert result == expected


class TestFlatten(TestCase):
    def test_example_1(self) -> None:
        xlst = [[[3.0, 0, 1], ["a", None, 2], range(3)]]
        expected = [3.0, 0, 1, "a", None, 2, 0, 1, 2]
        result = flatten(xlst)
        assert result == expected

    def test_example_2(self) -> None:
        xlst = [[range(0, 3), range(3, 6)], [range(6, 9), range(9, 12)]]

        expected = list(range(0, 12))
        result = flatten(xlst)
        assert result == expected

        expected = [list(range(0, 6)), list(range(6, 12))]
        result = flatten(xlst, 1, 2)
        assert result == expected

        expected = [range(0, 3), range(3, 6), range(6, 9), range(9, 12)]
        result = flatten(xlst, 0, 1)
        assert result == expected


if __name__ == "__main__":
    unittest.main()
