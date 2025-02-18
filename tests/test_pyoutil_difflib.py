#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.pyoutil.difflib import find_closest_in_list


class TestPyoutilMath(TestCase):
    def test_find(self) -> None:
        lst = ["a", "ghi", "abcd", "defff", "dff", ""]

        examples = [
            ("abc", lst, "abcd"),
            ("def", lst, "defff"),
            ("ghi", lst, "ghi"),
            ("", lst, ""),
            ("b", lst, "abcd"),
        ]

        for word_i, lst_i, expected_i in examples:
            result = find_closest_in_list(word_i, lst_i)
            assert result == expected_i


if __name__ == "__main__":
    unittest.main()
