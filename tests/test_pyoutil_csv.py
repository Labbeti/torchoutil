#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.pyoutil.csv import dump_csv


class TestCSV(TestCase):
    def test_find(self) -> None:
        examples = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        expected = "a,b\r\n1,2\r\n3,4\r\n"
        assert dump_csv(examples) == expected

        expected = "1,2\r\n3,4\r\n"
        assert dump_csv(examples, header=False) == expected


if __name__ == "__main__":
    unittest.main()
