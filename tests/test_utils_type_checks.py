#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.utils.type_checks import is_iterable_str


class TestTypeChecks(TestCase):
    def test_is_iterable_str_1(self) -> None:
        inputs = [
            ("a", True, False),
            (["a"], True, True),
            ([], True, True),
            (("a",), True, True),
            ((), True, True),
            (1.0, False, False),
        ]

        for x, expected_1, expected_2 in inputs:
            result_1 = is_iterable_str(x, accept_str=True)
            result_2 = is_iterable_str(x, accept_str=False)

            assert expected_1 == result_1
            assert expected_2 == result_2


if __name__ == "__main__":
    unittest.main()
